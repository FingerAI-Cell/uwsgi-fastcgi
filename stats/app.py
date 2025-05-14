import os
import time
import json
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
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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
LOG_CHECK_INTERVAL = int(os.environ.get('LOG_CHECK_INTERVAL', 10))  # 초 단위

# 데이터베이스 연결 문자열
DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Flask 애플리케이션 생성
app = Flask(__name__)
CORS(app)

# 파일 시스템 감시 설정
observer = None
log_processor = None
log_thread = None

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
    if observer and observer.is_alive():
        observer.stop()
        observer.join()
    if log_processor:
        log_processor.stop()

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
        
    def process_new_logs(self):
        """로그 파일에서 새로운 라인을 읽고 처리합니다."""
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
                        self.process_log_line(line.strip())
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
            
            # 필요한 데이터 추출
            endpoint = log_data.get('endpoint', '')
            method = log_data.get('method', 'GET')
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
    
    def run(self):
        """일정 간격으로 로그 파일 확인"""
        while self.should_run:
            self.process_new_logs()
            time.sleep(LOG_CHECK_INTERVAL)
    
    def stop(self):
        """처리 중지"""
        self.should_run = False

# 파일 시스템 이벤트 핸들러
class LogFileHandler(FileSystemEventHandler):
    def __init__(self, log_processor):
        self.log_processor = log_processor
        super().__init__()
    
    def on_modified(self, event):
        if event.src_path == self.log_processor.log_file_path:
            self.log_processor.process_new_logs()

def start_log_monitoring():
    """로그 모니터링 시작"""
    global log_processor, log_thread, observer
    
    # 이미 실행 중인 경우 중복 실행 방지
    if log_thread and log_thread.is_alive():
        logger.info("로그 모니터링이 이미 실행 중입니다.")
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
        logger.info("로그 처리 스레드 시작")
        
        # 파일 시스템 감시 시작
        event_handler = LogFileHandler(log_processor)
        observer = Observer()
        observer.schedule(event_handler, log_dir, recursive=False)
        observer.start()
        logger.info(f"로그 파일 감시 시작: {NGINX_LOG_PATH}")
    
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

# API 통계 로깅 엔드포인트 - Nginx에서 mirror 지시문을 통해 요청됨
@app.route('/api/log', methods=['GET', 'POST'])
def log_api():
    try:
        # 요청 정보 추출 (Nginx의 proxy_set_header에서 전달된 헤더 사용)
        endpoint = request.headers.get('X-Original-URI', '').split('?')[0]
        method = request.headers.get('X-Original-Method', request.method)
        status_code = int(request.headers.get('X-Response-Status', 200))
        response_time = float(request.headers.get('X-Response-Time', 0))
        user_agent = request.headers.get('User-Agent', '')
        ip_address = request.headers.get('X-Real-IP', request.remote_addr)
        
        # 요청 크기 계산
        request_size = len(request.get_data()) if request.get_data() else None
        
        # 비동기로 로깅 (실제로는 백그라운드 작업으로 처리하는 것이 좋음)
        log_api_call(endpoint, method, status_code, response_time, request_size, None, user_agent, ip_address)
        
        # 로깅 요청에는 204 No Content로 응답
        return "", 204
    except Exception as e:
        logger.error(f"로깅 중 오류 발생: {e}")
        return "", 204  # 오류가 발생해도 원래 요청에 영향을 주지 않도록 204 반환

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