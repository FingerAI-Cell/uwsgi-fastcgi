import os
import time
import json
import fcntl  # 파일 잠금을 위한 모듈 추가
import signal
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pymysql
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import threading
import json

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 데이터베이스 연결 설정
DB_HOST = os.environ.get('MYSQL_HOST', 'stats-db')
DB_PORT = int(os.environ.get('MYSQL_PORT', 3306))
DB_USER = os.environ.get('MYSQL_USER', 'stats_user')
DB_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'stats_password')
DB_NAME = os.environ.get('MYSQL_DATABASE', 'api_stats')

# 로그 파일 설정
NGINX_LOG_PATH = os.environ.get('NGINX_LOG_PATH', '/var/log/nginx/api-stats.log')
LOCK_FILE_PATH = os.environ.get('LOCK_FILE_PATH', '/tmp/stats_log_processor.lock')
LOG_CHECK_INTERVAL = int(os.environ.get('LOG_CHECK_INTERVAL', 5))  # 5초 간격으로 변경

# 데이터베이스 연결 문자열
DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Flask 애플리케이션 생성
app = Flask(__name__)
CORS(app)

# 파일 시스템 감시 설정 변경
log_processor = None
log_thread = None
lock_file = None
has_lock = False

# 애플리케이션 초기화 함수
def init_app(app):
    # SQLAlchemy 엔진 생성
    global engine
    try:
        engine = create_engine(DATABASE_URI)
        logger.info("데이터베이스 연결 성공")
    except Exception as e:
        logger.error(f"데이터베이스 연결 실패: {e}")
        engine = None
    
    # 로그 모니터링 시작
    start_log_monitoring()

# 애플리케이션 종료 시 리소스 정리
def cleanup():
    global lock_file, has_lock
    if log_processor:
        log_processor.stop()
    # 파일 잠금 해제
    if lock_file and has_lock:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            logger.info("파일 잠금 해제 완료")
        except Exception as e:
            logger.error(f"파일 잠금 해제 중 오류: {e}")

# 종료 시그널 핸들러
def signal_handler(sig, frame):
    logger.info("종료 시그널 수신, 리소스 정리 중...")
    cleanup()
    os._exit(0)

# 종료 시그널 처리
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# SQLAlchemy 엔진 생성
engine = None

# 데이터베이스 연결 확인
def check_db_connection():
    if engine is None:
        return False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as e:
        logger.error(f"데이터베이스 연결 확인 실패: {e}")
        return False

# 로깅 함수
def log_api_call(endpoint, method, status_code, response_time, request_size=None, response_size=None, user_agent=None, ip_address=None):
    if engine is None:
        logger.error("데이터베이스 엔진이 초기화되지 않았습니다.")
        return
    
    try:
        query = text("""
            INSERT INTO api_calls (endpoint, method, status_code, response_time, request_size, response_size, user_agent, ip_address)
            VALUES (:endpoint, :method, :status_code, :response_time, :request_size, :response_size, :user_agent, :ip_address)
        """)
        
        with engine.connect() as conn:
            conn.execute(query, {
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'response_time': response_time,
                'request_size': request_size,
                'response_size': response_size,
                'user_agent': user_agent,
                'ip_address': ip_address
            })
            conn.commit()
            
        logger.info(f"API 호출 로깅 성공: {endpoint} - {status_code}")
    except SQLAlchemyError as e:
        logger.error(f"API 호출 로깅 실패: {e}")

# 로그 처리 클래스
class LogProcessor:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.last_position = 0
        self.should_run = True
        self.initialized = False
        # 최근 처리한 로그 ID를 저장하기 위한 세트
        self.processed_logs = set()
        
    def initialize(self):
        """
        로그 파일이 존재하면 파일의 끝으로 이동하여 새로운 로그만 처리하도록 초기화합니다.
        이미 처리된 기존 로그를 다시 처리하지 않도록 합니다.
        """
        if os.path.exists(self.log_file_path):
            try:
                with open(self.log_file_path, 'r') as f:
                    # 파일의 끝으로 이동
                    f.seek(0, 2)  # 2는 파일의 끝에서부터의 오프셋을 의미
                    self.last_position = f.tell()
                self.initialized = True
                logger.info(f"로그 파일 초기화 완료: {self.log_file_path}, 위치: {self.last_position}")
            except Exception as e:
                logger.error(f"로그 파일 초기화 중 오류: {e}")
    
    def process_new_logs(self):
        """로그 파일에서 새로운 라인을 읽고 처리합니다."""
        if not self.initialized:
            self.initialize()
            
        if not os.path.exists(self.log_file_path):
            logger.warning(f"로그 파일이 없습니다: {self.log_file_path}")
            return
        
        try:
            with open(self.log_file_path, 'r') as f:
                # 마지막으로 읽은 위치로 이동
                f.seek(self.last_position)
                
                # 새 라인 읽기
                new_lines = f.readlines()
                
                # 현재 위치 저장
                self.last_position = f.tell()
                
                # 새 라인 처리
                for line in new_lines:
                    try:
                        line = line.strip()
                        # 빈 라인 무시
                        if not line:
                            continue
                            
                        # 중복 로그 방지를 위해 해시값 계산
                        log_hash = hash(line)
                        if log_hash in self.processed_logs:
                            logger.debug(f"이미 처리된 로그 무시: {line[:50]}...")
                            continue
                            
                        self.process_log_line(line)
                        
                        # 처리된 로그 해시 저장 (최대 1000개까지만 유지)
                        self.processed_logs.add(log_hash)
                        if len(self.processed_logs) > 1000:
                            self.processed_logs.pop()
                    except Exception as e:
                        logger.error(f"로그 라인 처리 중 오류: {e}")
        
        except Exception as e:
            logger.error(f"로그 파일 처리 중 오류: {e}")
    
    def process_log_line(self, line):
        """JSON 형식의 로그 라인을 파싱하고 데이터베이스에 저장합니다."""
        if not line:
            return
            
        try:
            # JSON 파싱
            log_data = json.loads(line)
            
            # 로그 중복 방지를 위한 검사
            endpoint = log_data.get('endpoint', '')
            method = log_data.get('method', 'GET')
            time_str = log_data.get('time', '')
            
            # 이미 DB에 저장된 로그인지 확인
            if self.is_log_already_saved(endpoint, method, time_str):
                logger.debug(f"이미 저장된 로그 무시: {endpoint} {method} {time_str}")
                return
            
            # 필요한 데이터 추출
            status = int(log_data.get('status', 200))
            response_time = float(log_data.get('response_time', 0))
            body_bytes_sent = int(log_data.get('body_bytes_sent', 0))
            remote_addr = log_data.get('remote_addr', '')
            user_agent = log_data.get('user_agent', '')
            
            # 데이터베이스에 저장
            log_api_call(
                endpoint=endpoint,
                method=method,
                status_code=status,
                response_time=response_time,
                request_size=None,  # 로그에서는 요청 크기를 알 수 없음
                response_size=body_bytes_sent,
                user_agent=user_agent,
                ip_address=remote_addr
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}, 라인: {line[:100]}")
        except Exception as e:
            logger.error(f"로그 처리 중 오류: {e}")
    
    def is_log_already_saved(self, endpoint, method, time_str):
        """이미 데이터베이스에 저장된 로그인지 확인합니다."""
        if engine is None or not time_str:
            return False
            
        try:
            # ISO 8601 형식의 시간문자열을 MySQL datetime 형식으로 변환
            log_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            log_time_start = log_time - timedelta(seconds=1)
            log_time_end = log_time + timedelta(seconds=1)
            
            query = text("""
                SELECT COUNT(*) as count FROM api_calls 
                WHERE endpoint = :endpoint 
                AND method = :method 
                AND created_at BETWEEN :start_time AND :end_time
            """)
            
            with engine.connect() as conn:
                result = conn.execute(query, {
                    'endpoint': endpoint,
                    'method': method,
                    'start_time': log_time_start,
                    'end_time': log_time_end
                }).fetchone()
                
                # 이미 저장된 로그가 있으면 True 반환
                return result and result[0] > 0
                
        except Exception as e:
            logger.error(f"로그 중복 확인 중 오류: {e}")
            return False
        
    def run(self):
        """일정 간격으로 로그 파일 확인"""
        # 시작할 때 초기화
        if not self.initialized:
            self.initialize()
            
        while self.should_run:
            self.process_new_logs()
            time.sleep(LOG_CHECK_INTERVAL)
    
    def stop(self):
        """처리 중지"""
        self.should_run = False

def acquire_lock():
    """
    파일 잠금을 획득하여 여러 워커가 동시에 로그를 처리하지 않도록 합니다.
    잠금을 획득하면 True, 그렇지 않으면 False를 반환합니다.
    """
    global lock_file, has_lock
    
    try:
        # 잠금 파일 생성 또는 열기
        lock_file = open(LOCK_FILE_PATH, 'w')
        
        # 비차단 잠금 시도
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        
        # 잠금 획득 성공
        has_lock = True
        logger.info("로그 처리기 잠금 획득 성공")
        return True
        
    except IOError:
        # 이미 다른 프로세스가 잠금을 보유하고 있음
        if lock_file:
            lock_file.close()
            lock_file = None
        logger.info("다른 프로세스가 이미 로그 처리 중입니다.")
        return False
    except Exception as e:
        logger.error(f"파일 잠금 획득 중 오류: {e}")
        if lock_file:
            lock_file.close()
            lock_file = None
        return False

def start_log_monitoring():
    """로그 모니터링 시작"""
    global log_processor, log_thread
    
    # 이미 실행 중인 경우 중복 실행 방지
    if log_thread and log_thread.is_alive():
        logger.info("로그 모니터링이 이미 실행 중입니다.")
        return
    
    # 잠금 획득 시도
    if not acquire_lock():
        logger.info("이 워커는 로그 처리를 건너뜁니다.")
        return
    
    # 로그 디렉토리가 존재하는지 확인
    log_dir = os.path.dirname(NGINX_LOG_PATH)
    if not os.path.exists(log_dir):
        logger.warning(f"로그 디렉토리가 없습니다: {log_dir}")
        return
    
    try:
        # 로그 처리기 초기화
        log_processor = LogProcessor(NGINX_LOG_PATH)
        
        # 로그 처리 스레드 시작
        log_thread = threading.Thread(target=log_processor.run, daemon=True)
        log_thread.start()
        logger.info("로그 처리 스레드 시작, 주기적 확인 간격: {}초".format(LOG_CHECK_INTERVAL))
    
    except Exception as e:
        logger.error(f"로그 모니터링 시작 중 오류: {e}")

# 상태 확인 엔드포인트
@app.route('/health', methods=['GET'])
def health_check():
    db_status = "connected" if check_db_connection() else "disconnected"
    return jsonify({
        "status": "healthy",
        "service": "stats-service",
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    })

# 통계 조회 API
@app.route('/api/stats', methods=['GET'])
def get_stats():
    if engine is None:
        return jsonify({"error": "데이터베이스 연결이 없습니다."}), 500
    
    try:
        # 기간 파라미터
        days = request.args.get('days', default=7, type=int)
        if days <= 0:
            days = 7
        
        # 시작일과 종료일 계산
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # 일별 통계 쿼리
        daily_query = text("""
            SELECT 
                endpoint, 
                date, 
                total_calls,
                success_calls,
                error_calls,
                avg_response_time,
                max_response_time,
                min_response_time
            FROM daily_stats
            WHERE date BETWEEN :start_date AND :end_date
            ORDER BY date DESC, total_calls DESC
        """)
        
        # 엔드포인트별 합계 쿼리
        endpoint_query = text("""
            SELECT 
                endpoint,
                SUM(total_calls) as total_calls,
                SUM(success_calls) as success_calls,
                SUM(error_calls) as error_calls,
                AVG(avg_response_time) as avg_response_time
            FROM daily_stats
            WHERE date BETWEEN :start_date AND :end_date
            GROUP BY endpoint
            ORDER BY total_calls DESC
        """)
        
        # 최근 호출 쿼리
        recent_query = text("""
            SELECT 
                endpoint,
                method,
                status_code,
                response_time,
                created_at
            FROM api_calls
            ORDER BY created_at DESC
            LIMIT 50
        """)
        
        # 데이터 조회
        with engine.connect() as conn:
            daily_stats = [dict(row) for row in conn.execute(daily_query, {"start_date": start_date, "end_date": end_date})]
            endpoint_stats = [dict(row) for row in conn.execute(endpoint_query, {"start_date": start_date, "end_date": end_date})]
            recent_calls = [dict(row) for row in conn.execute(recent_query)]
            
            # datetime 객체를 JSON 직렬화 가능한 형태로 변환
            for row in daily_stats:
                if 'date' in row:
                    row['date'] = row['date'].isoformat() if row['date'] else None
                    
            for row in recent_calls:
                if 'created_at' in row:
                    row['created_at'] = row['created_at'].isoformat() if row['created_at'] else None
        
        return jsonify({
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "endpoint_stats": endpoint_stats,
            "daily_stats": daily_stats,
            "recent_calls": recent_calls
        })
        
    except SQLAlchemyError as e:
        logger.error(f"통계 조회 중 오류: {e}")
        return jsonify({"error": "데이터베이스 쿼리 실행 중 오류가 발생했습니다."}), 500
    except Exception as e:
        logger.error(f"통계 조회 중 예상치 못한 오류: {e}")
        return jsonify({"error": "통계 조회 중 오류가 발생했습니다."}), 500

# 대시보드 페이지
@app.route('/', methods=['GET'])
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API 통계 대시보드</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }
            h1 { color: #2c3e50; }
            .container { max-width: 1200px; margin: 0 auto; }
            .stats-container { margin-top: 20px; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .card { background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }
            .summary { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }
            .summary-card { flex: 1; min-width: 200px; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); background-color: #f8f9fa; }
            .summary-card h3 { margin-top: 0; color: #2c3e50; }
            .summary-card p { font-size: 24px; font-weight: bold; margin: 5px 0; }
            .tabs { display: flex; margin-bottom: 15px; border-bottom: 1px solid #ddd; }
            .tab { padding: 10px 15px; cursor: pointer; }
            .tab.active { border-bottom: 2px solid #2c3e50; font-weight: bold; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .loading { text-align: center; padding: 50px; font-size: 18px; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>API 통계 대시보드</h1>
            
            <div class="card">
                <div class="filter">
                    <label for="period">기간: </label>
                    <select id="period" onchange="loadStats()">
                        <option value="7">최근 7일</option>
                        <option value="14">최근 14일</option>
                        <option value="30">최근 30일</option>
                    </select>
                    <button onclick="loadStats()">새로고침</button>
                </div>
            </div>
            
            <div id="summary" class="summary">
                <div class="loading">데이터 로딩 중...</div>
            </div>
            
            <div class="card">
                <div class="tabs">
                    <div class="tab active" onclick="showTab('endpoint-stats')">엔드포인트별 통계</div>
                    <div class="tab" onclick="showTab('daily-stats')">일별 통계</div>
                    <div class="tab" onclick="showTab('recent-calls')">최근 호출</div>
                </div>
                
                <div id="endpoint-stats" class="tab-content active">
                    <div class="loading">데이터 로딩 중...</div>
                </div>
                
                <div id="daily-stats" class="tab-content">
                    <div class="loading">데이터 로딩 중...</div>
                </div>
                
                <div id="recent-calls" class="tab-content">
                    <div class="loading">데이터 로딩 중...</div>
                </div>
            </div>
        </div>
        
        <script>
            // 탭 전환 함수
            function showTab(tabId) {
                // 모든 탭 비활성화
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                
                // 선택한 탭 활성화
                document.querySelector(`.tab[onclick="showTab('${tabId}')"]`).classList.add('active');
                document.getElementById(tabId).classList.add('active');
            }
            
            // 통계 데이터 로드 함수
            async function loadStats() {
                const days = document.getElementById('period').value;
                
                try {
                    const response = await fetch(`/api/stats?days=${days}`);
                    if (!response.ok) {
                        throw new Error('통계 데이터를 가져오는데 실패했습니다.');
                    }
                    
                    const data = await response.json();
                    
                    // 요약 정보 업데이트
                    updateSummary(data);
                    
                    // 엔드포인트별 통계 테이블 업데이트
                    updateEndpointStats(data.endpoint_stats);
                    
                    // 일별 통계 테이블 업데이트
                    updateDailyStats(data.daily_stats);
                    
                    // 최근 호출 테이블 업데이트
                    updateRecentCalls(data.recent_calls);
                    
                } catch (error) {
                    console.error('Error:', error);
                    document.querySelectorAll('.loading').forEach(el => {
                        el.textContent = '데이터를 가져오는데 실패했습니다.';
                    });
                }
            }
            
            // 요약 정보 업데이트 함수
            function updateSummary(data) {
                let totalCalls = 0;
                let successCalls = 0;
                let errorCalls = 0;
                let avgResponseTime = 0;
                let endpointCount = 0;
                
                if (data.endpoint_stats && data.endpoint_stats.length > 0) {
                    endpointCount = data.endpoint_stats.length;
                    
                    // 전체 통계 계산
                    data.endpoint_stats.forEach(endpoint => {
                        totalCalls += endpoint.total_calls;
                        successCalls += endpoint.success_calls;
                        errorCalls += endpoint.error_calls;
                        avgResponseTime += endpoint.avg_response_time * endpoint.total_calls;
                    });
                    
                    // 평균 응답 시간 계산
                    avgResponseTime = totalCalls > 0 ? avgResponseTime / totalCalls : 0;
                }
                
                // 요약 카드 생성
                const summaryHTML = `
                    <div class="summary-card">
                        <h3>총 호출 수</h3>
                        <p>${totalCalls.toLocaleString()}</p>
                    </div>
                    <div class="summary-card">
                        <h3>성공률</h3>
                        <p>${totalCalls > 0 ? (successCalls / totalCalls * 100).toFixed(2) : 0}%</p>
                    </div>
                    <div class="summary-card">
                        <h3>평균 응답 시간</h3>
                        <p>${avgResponseTime.toFixed(3)}초</p>
                    </div>
                    <div class="summary-card">
                        <h3>API 엔드포인트 수</h3>
                        <p>${endpointCount}</p>
                    </div>
                `;
                
                document.getElementById('summary').innerHTML = summaryHTML;
            }
            
            // 엔드포인트별 통계 테이블 업데이트 함수
            function updateEndpointStats(endpoints) {
                if (!endpoints || endpoints.length === 0) {
                    document.getElementById('endpoint-stats').innerHTML = '<p>데이터가 없습니다.</p>';
                    return;
                }
                
                let tableHTML = `
                    <table>
                        <thead>
                            <tr>
                                <th>엔드포인트</th>
                                <th>총 호출 수</th>
                                <th>성공</th>
                                <th>실패</th>
                                <th>성공률</th>
                                <th>평균 응답 시간(초)</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                endpoints.forEach(endpoint => {
                    const successRate = endpoint.total_calls > 0 ? (endpoint.success_calls / endpoint.total_calls * 100).toFixed(2) : 0;
                    
                    tableHTML += `
                        <tr>
                            <td>${endpoint.endpoint}</td>
                            <td>${endpoint.total_calls.toLocaleString()}</td>
                            <td>${endpoint.success_calls.toLocaleString()}</td>
                            <td>${endpoint.error_calls.toLocaleString()}</td>
                            <td>${successRate}%</td>
                            <td>${endpoint.avg_response_time.toFixed(3)}</td>
                        </tr>
                    `;
                });
                
                tableHTML += `
                        </tbody>
                    </table>
                `;
                
                document.getElementById('endpoint-stats').innerHTML = tableHTML;
            }
            
            // 일별 통계 테이블 업데이트 함수
            function updateDailyStats(dailyStats) {
                if (!dailyStats || dailyStats.length === 0) {
                    document.getElementById('daily-stats').innerHTML = '<p>데이터가 없습니다.</p>';
                    return;
                }
                
                let tableHTML = `
                    <table>
                        <thead>
                            <tr>
                                <th>날짜</th>
                                <th>엔드포인트</th>
                                <th>총 호출 수</th>
                                <th>성공</th>
                                <th>실패</th>
                                <th>평균 응답 시간(초)</th>
                                <th>최대 응답 시간(초)</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                dailyStats.forEach(stat => {
                    const date = new Date(stat.date).toLocaleDateString();
                    
                    tableHTML += `
                        <tr>
                            <td>${date}</td>
                            <td>${stat.endpoint}</td>
                            <td>${stat.total_calls.toLocaleString()}</td>
                            <td>${stat.success_calls.toLocaleString()}</td>
                            <td>${stat.error_calls.toLocaleString()}</td>
                            <td>${stat.avg_response_time.toFixed(3)}</td>
                            <td>${stat.max_response_time.toFixed(3)}</td>
                        </tr>
                    `;
                });
                
                tableHTML += `
                        </tbody>
                    </table>
                `;
                
                document.getElementById('daily-stats').innerHTML = tableHTML;
            }
            
            // 최근 호출 테이블 업데이트 함수
            function updateRecentCalls(recentCalls) {
                if (!recentCalls || recentCalls.length === 0) {
                    document.getElementById('recent-calls').innerHTML = '<p>데이터가 없습니다.</p>';
                    return;
                }
                
                let tableHTML = `
                    <table>
                        <thead>
                            <tr>
                                <th>시간</th>
                                <th>엔드포인트</th>
                                <th>메서드</th>
                                <th>상태 코드</th>
                                <th>응답 시간(초)</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                recentCalls.forEach(call => {
                    const date = new Date(call.created_at).toLocaleString();
                    const statusClass = call.status_code >= 400 ? 'error' : 'success';
                    
                    tableHTML += `
                        <tr>
                            <td>${date}</td>
                            <td>${call.endpoint}</td>
                            <td>${call.method}</td>
                            <td class="${statusClass}">${call.status_code}</td>
                            <td>${call.response_time.toFixed(3)}</td>
                        </tr>
                    `;
                });
                
                tableHTML += `
                        </tbody>
                    </table>
                `;
                
                document.getElementById('recent-calls').innerHTML = tableHTML;
            }
            
            // 페이지 로드 시 통계 데이터 가져오기
            document.addEventListener('DOMContentLoaded', loadStats);
        </script>
    </body>
    </html>
    """

# 앱 초기화
init_app(app)

# 메인 실행 코드
if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        cleanup()