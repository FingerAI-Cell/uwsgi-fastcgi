from flask import Flask, send_file, request, jsonify, Response
from pymilvus import Collection
from dotenv import load_dotenv
from src import EnvManager, InteractManager
from src.pipe import InteractManager as PipeInteractManager  # 명시적으로 pipe.py의 InteractManager 임포트
import logging
import json 
import os 
import time
from flask.cli import with_appcontext
import click
import atexit
import torch
import datetime
import sys
import signal
import concurrent.futures
import threading
import traceback
import psutil
import hashlib

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 로그 디렉토리 확인 및 생성
log_dir = "/var/log/rag" if os.path.exists("/var/log/rag") else "logs"
os.makedirs(log_dir, exist_ok=True)
print(f"로그 디렉토리: {log_dir}")

# 메인 로거 설정
logger = logging.getLogger("rag-backend")
logger.setLevel(logging.INFO)
logger.handlers = []  # 기존 핸들러 제거

# 스트림 핸들러 추가
stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

# 파일 핸들러 추가
file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.propagate = False  # 루트 로거로 전파 방지

# 시간 로깅 전용 로거 설정
timing_logger = logging.getLogger('timing')
timing_logger.setLevel(logging.INFO)
timing_logger.handlers = []  # 기존 핸들러 제거
timing_handler = logging.FileHandler(os.path.join(log_dir, 'timing.log'))
timing_formatter = logging.Formatter('%(asctime)s - %(message)s')
timing_handler.setFormatter(timing_formatter)
timing_logger.addHandler(timing_handler)
timing_logger.propagate = False  # 다른 로거로 전파 방지

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# FastCGI 응답 형식 설정
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['PREFERRED_URL_SCHEME'] = 'http'

load_dotenv()
args = dict()
args['config_path'] = "./config"
args['llm_config'] = "llm_config.json"
args['db_config'] = "db_config.json"
args['collection_name'] = "congress"
args['ip_addr'] = os.getenv('MILVUS_HOST')   

env_manager = EnvManager(args)
env_manager.set_processors()
emb_model = env_manager.set_emb_model()
milvus_data, milvus_meta = env_manager.set_vectordb()
milvus_db = env_manager.milvus_db
interact_manager = InteractManager(data_p=env_manager.data_p, vectorenv=milvus_db, vectordb=milvus_data, emb_model=emb_model)

# 자주 사용하는 컬렉션을 미리 로드하는 함수
def load_common_collections():
    """자주 사용하는 컬렉션을 미리 로드합니다."""
    logger.info("Preloading common collections...")
    
    try:
        # 사용 가능한 모든 컬렉션 얻기
        available_collections = milvus_db.get_list_collection()
        logger.info(f"Available collections: {available_collections}")
        
        # 자주 사용하는 컬렉션 목록 정의 (필요에 따라 수정)
        common_collections = ["news", "congress"]
        # 실제 존재하는 컬렉션만 처리
        collections_to_load = [c for c in common_collections if c in available_collections]
        
        # 존재하는 컬렉션 없음 시 종료
        if not collections_to_load:
            logger.warning("No common collections found to preload")
            return
        
        logger.info(f"Will preload these collections: {collections_to_load}")
        
        # 컬렉션 로드
        for collection_name in collections_to_load:
            try:
                # 각 컬렉션 로드 시간 측정
                start_time = time.time()
                collection = interact_manager.get_collection(collection_name)
                load_time = time.time() - start_time
                logger.info(f"Preloaded collection '{collection_name}' in {load_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to preload collection '{collection_name}': {str(e)}")
        
        logger.info("Finished preloading collections")
        
    except Exception as e:
        logger.error(f"Error during collection preloading: {str(e)}")

# 현재 프로세스의 메모리 사용량을 MB 단위로 반환하는 함수
def get_memory_usage():
    """
    현재 프로세스의 메모리 사용량을 메가바이트(MB) 단위로 반환합니다.
    
    Returns:
        float: 메모리 사용량 (MB)
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        # RSS(Resident Set Size)를 MB로 변환
        memory_mb = memory_info.rss / (1024 * 1024)
        return memory_mb
    except Exception as e:
        logger.warning(f"메모리 사용량 조회 실패: {str(e)}")
        return 0.0  # 실패 시 0 반환

# 앱 종료 시 정리 작업
def cleanup_on_exit():
    """애플리케이션 종료 시 정리 작업을 수행합니다."""
    logger.info("Application shutting down, performing cleanup...")
    try:
        # 임베딩 배치 워커 종료
        try:
            from src.pipe import InteractManager
            logger.info("임베딩 배치 처리 워커 중지 중...")
            InteractManager.stop_embedding_worker()
        except Exception as emb_worker_error:
            logger.error(f"임베딩 배치 워커 정리 실패: {str(emb_worker_error)}")
        
        # 글로벌 배치 워커 스레드 정리
        try:
            from src.pipe import InteractManager
            # 남은 배치 데이터 처리
            logger.info("남은 배치 데이터 처리 중...")
            InteractManager.flush_all_batches()
            # 배치 워커 중지
            logger.info("배치 처리 워커 중지 중...")
            InteractManager.stop_batch_worker()
        except Exception as batch_error:
            logger.error(f"배치 워커 정리 실패: {str(batch_error)}")
        
        # 캐시된 컬렉션 정리
        if hasattr(interact_manager, 'loaded_collections'):
            logger.info(f"Clearing {len(interact_manager.loaded_collections)} cached collections")
            interact_manager.loaded_collections.clear()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            try:
                # 안전하게 GPU 메모리 정리
                logger.info("Safely clearing GPU memory...")
                # 명시적으로 모델 정리
                if hasattr(emb_model, 'model'):
                    try:
                        emb_model.model = None
                    except:
                        pass
                
                # 가비지 컬렉션 실행
                import gc
                gc.collect()
                
                # 조심스럽게 캐시 비우기 시도
                try:
                    torch.cuda.empty_cache()
                    logger.info("GPU memory cache cleared")
                except Exception as e:
                    logger.error(f"Could not clear GPU cache: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to clear GPU memory: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        import traceback
        logger.error(f"Cleanup error details: {traceback.format_exc()}")
    
    logger.info("Cleanup complete")

# 종료 핸들러 등록
atexit.register(cleanup_on_exit)

# CLI 명령 대신 앱 시작 시점에 실행할 초기화 함수 등록
@click.command("load-collections")
@with_appcontext
def load_collections_command():
    """CLI 명령어로 컬렉션 로드를 수행합니다."""
    load_common_collections()
    click.echo("Collections loaded!")

# 앱 시작 시 초기화 함수 실행
# Flask 2.0 이상에서 before_first_request 대신 권장되는 방식
with app.app_context():
    # 앱이 시작될 때 컬렉션 로드
    load_common_collections()
    
    # 임베딩 배치 처리 워커 시작
    from src.pipe import InteractManager
    InteractManager.start_embedding_worker()
    logger.info("임베딩 배치 처리 워커 시작됨")

# app.cli에 명령 추가 (Flask CLI에서 실행 가능)
app.cli.add_command(load_collections_command)

# FastCGI 응답 헤더 설정
@app.after_request
def add_header(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

@app.route('/rag/data/show', methods=['GET'])
def show_data():
    print(f"Search results")
    '''
    "collection_name": "news"   # news, description, etc ... 
    '''
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({"error": "collection_name is required"}), 400
    
    try:
        assert Collection(collection_name)
    except: 
        response_data = {"error": "유효한 Collection 이름을 입력해야 합니다.", "collection list": milvus_db.get_list_collection()}
        return Response(json.dumps(response_data, ensure_ascii=False), content_type="application/json; charset=utf-8")
    
    milvus_db.get_collection_info(collection_name)
    milvus_db.get_partition_info(collection_name)               
    return jsonify({
        "schema": milvus_db.collection_schema.to_dict(),
        "partition_names": milvus_db.partition_names,
        "partition_nums": milvus_db.partition_entities_num,
    }), 200

@app.route('/rag/search', methods=['POST'])
def search_data():
    # 전체 API 실행 시간 측정 시작
    start_time = time.time()
    
    # JSON 데이터 받기
    request_data = request.get_json()
    if not request_data:
        return jsonify({
            "result_code": "F000000",
            "message": "JSON 데이터가 필요합니다.",
            "search_result": None
        }), 400

    # 기본 검색 파라미터
    query_text = request_data.get('query_text')
    top_k = request_data.get('top_k', 5)
    domains = request_data.get('domains', [])  # 도메인 리스트

    # 추가 필터링 파라미터
    author = request_data.get('author')  # 작성자 필터
    start_date = request_data.get('start_date')  # YYYYMMDD 형식
    end_date = request_data.get('end_date')      # YYYYMMDD 형식
    title_query = request_data.get('title')      # 제목 검색
    info_filter = request_data.get('info_filter') # info 필터 조건
    tags_filter = request_data.get('tags_filter') # tags 필터 조건

    logger.info(f"[TIMING] 검색 요청 시작: query='{query_text}', top_k={top_k}, domains={domains}")

    if not query_text:
        return jsonify({
            "result_code": "F000001",
            "message": "검색어(query_text)는 필수 입력값입니다.",
            "search_result": None
        }), 400

    try:
        top_k = int(top_k)
    except ValueError:
        return jsonify({
            "result_code": "F000002",
            "message": "top_k 값은 숫자여야 합니다.",
            "search_result": None
        }), 400

    # 날짜 형식 검증
    if start_date or end_date:
        date_error = None
        if start_date:
            if not (len(start_date) == 8 and start_date.isdigit() and 
                   1900 <= int(start_date[:4]) <= 2100 and 
                   1 <= int(start_date[4:6]) <= 12 and 
                   1 <= int(start_date[6:]) <= 31):
                date_error = "시작 날짜가 올바른 형식(YYYYMMDD)이 아닙니다."
        if end_date:
            if not (len(end_date) == 8 and end_date.isdigit() and 
                   1900 <= int(end_date[:4]) <= 2100 and 
                   1 <= int(end_date[4:6]) <= 12 and 
                   1 <= int(end_date[6:]) <= 31):
                date_error = "종료 날짜가 올바른 형식(YYYYMMDD)이 아닙니다."
        if start_date and end_date and start_date > end_date:
            date_error = "시작 날짜가 종료 날짜보다 늦을 수 없습니다."
        
        if date_error:
            return jsonify({
                "result_code": "F000006",
                "message": date_error,
                "search_result": None
            }), 400
    
    validation_time = time.time()
    logger.info(f"[TIMING] 파라미터 검증 완료: {(validation_time - start_time):.4f}초")
    
    # 도메인 유효성 검증
    available_collections = milvus_db.get_list_collection()
    if domains:
        invalid_domains = [d for d in domains if d not in available_collections]
        if invalid_domains:
            return jsonify({
                "result_code": "F000007",
                "message": f"유효하지 않은 도메인이 포함되어 있습니다: {', '.join(invalid_domains)}",
                "available_domains": available_collections,
                "search_result": None
            }), 400
    
    # 필터 조건 파싱
    filter_conditions = {}
    if domains:
        filter_conditions['domains'] = domains
    if author:
        filter_conditions['author'] = author
    if start_date or end_date:
        filter_conditions['date_range'] = {'start': start_date, 'end': end_date}
    if title_query:
        filter_conditions['title'] = title_query
    if info_filter:
        # JSON 파싱이 필요 없음 (이미 JSON 객체로 받음)
        filter_conditions['info'] = info_filter
    if tags_filter:
        # JSON 파싱이 필요 없음 (이미 JSON 객체로 받음)
        filter_conditions['tags'] = tags_filter
    
    filter_time = time.time()
    logger.info(f"[TIMING] 필터 조건 준비 완료: {(filter_time - validation_time):.4f}초")
    
    try:
        # 각 도메인별 검색 결과 수집
        all_results = []
        search_time_details = {}
        searched_domains = domains if domains else [available_collections[0]]  # 도메인이 지정되지 않으면 첫 번째 도메인만 사용
        
        domain_start_time = time.time()
        logger.info(f"[TIMING] 도메인 검색 시작: {searched_domains}")
        
        for domain in searched_domains:
            domain_timer = time.time()
            logger.info(f"[TIMING] 도메인 '{domain}' 검색 시작")
            
            domain_filter_conditions = dict(filter_conditions)
            domain_filter_conditions['domain'] = domain
            
            # 검색 실행 시간 측정
            search_start = time.time()
            results = interact_manager.retrieve_data(
                query_text, 
                top_k, 
                filter_conditions=domain_filter_conditions
            )
            search_end = time.time()
            
            search_time = search_end - search_start
            search_time_details[domain] = {
                "time": search_time,
                "results_count": len(results)
            }
            logger.info(f"[TIMING] 도메인 '{domain}' 검색 완료: {search_time:.4f}초, 결과 {len(results)}개")
            
            # 도메인 정보 추가
            for result in results:
                result['domain'] = domain
            
            all_results.extend(results)
            
            domain_end_timer = time.time()
            logger.info(f"[TIMING] 도메인 '{domain}' 전체 처리: {(domain_end_timer - domain_timer):.4f}초")
        
        domain_end_time = time.time()
        logger.info(f"[TIMING] 모든 도메인 검색 완료: {(domain_end_time - domain_start_time):.4f}초, 총 결과 {len(all_results)}개")
        
        # 결과 정렬 시작
        sort_start = time.time()
        
        # 전체 결과를 score 기준으로 재정렬하고 top_k개만 선택
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = all_results[:top_k]
        
        sort_end = time.time()
        logger.info(f"[TIMING] 결과 정렬 완료: {(sort_end - sort_start):.4f}초")
        
        # 응답 포맷팅 시작
        format_start = time.time()
        
        # 각 결과에서 불필요한 필드 제거 및 정리
        cleaned_results = []
        for result in final_results:
            cleaned_result = {
                "doc_id": result.get('doc_id'),
                "raw_doc_id": result.get('raw_doc_id'),
                "passage_id": result.get('passage_id'),
                "domain": result.get('domain'),
                "title": result.get('title'),
                "author": result.get('author'),
                "text": result.get('text'),
                "info": result.get('info'),
                "tags": result.get('tags'),
                "score": result.get('score')
            }
            cleaned_results.append(cleaned_result)
        
        format_end = time.time()
        logger.info(f"[TIMING] 응답 포맷팅 완료: {(format_end - format_start):.4f}초")
        
        # 응답 생성
        response_data = {
            "result_code": "F000000",
            "message": "검색이 성공적으로 완료되었습니다.",
            "search_params": {
                "query_text": query_text,
                "top_k": top_k,
                "domains": searched_domains,
                "filters": filter_conditions
            },
            "total_results": len(all_results),
            "returned_results": len(cleaned_results),
            "search_result": cleaned_results,
            "performance": {
                "total_time": time.time() - start_time,
                "domain_details": search_time_details
            }
        }
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"[TIMING] 검색 API 완료: 총 {total_time:.4f}초 소요")
        
        return Response(json.dumps(response_data, ensure_ascii=False), 
                      content_type="application/json; charset=utf-8")
                      
    except Exception as e:
        error_time = time.time()
        total_time = error_time - start_time
        logger.error(f"[TIMING] 검색 오류 발생: {total_time:.4f}초 소요, 오류: {str(e)}")
        return jsonify({
            "result_code": "F000005",
            "message": f"검색 중 오류가 발생했습니다: {str(e)}",
            "search_result": None,
            "performance": {
                "total_time": total_time
            }
        }), 500

@app.route('/rag/insert', methods=['POST'])
def insert_data():
    '''
    {
        "documents": [
            {
                "domain": "news",
                "title": "메타버스 뉴스",
                "author": "삼성전자",
                "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다...",
                "info": {
                    "press_num": "비즈니스 워치",
                    "url": "http://example.com/news/1"
                },
                "tags": {
                    "date": "20240315",
                    "user": "admin"
                }
            }
        ],
        "ignore": true
    }
    '''
    # API 전체 실행 시간 측정 시작
    api_start_time = time.time()
    logger.info("=== INSERT API START ===")
    
    # insert API 전용 로거 설정
    insert_logger = logging.getLogger('insert')
    if not insert_logger.handlers:
        log_path = os.path.join(log_dir, 'insert.log')
        insert_handler = logging.FileHandler(log_path)
        insert_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        insert_handler.setFormatter(insert_formatter)
        insert_logger.setLevel(logging.INFO)
        insert_logger.addHandler(insert_handler)
        insert_logger.propagate = False  # 다른 로거로 전파 방지
    
    insert_logger.info("=== INSERT API START ===")
    
    # 배치 처리를 위한 공유 변수 초기화
    processed_chunks = []  # 처리된 청크를 담을 리스트
    chunks_lock = threading.Lock()  # 스레드 안전성을 위한 락
    batch_size = int(os.getenv('BATCH_SIZE', '300'))  # 배치 크기 설정
    
    # 배치 삽입 함수 정의
    def insert_batch(batch, domain):
        try:
            if not batch:
                return
                
            batch_insert_start = time.time()
            interact_manager.batch_insert_data(domain, batch)
            batch_insert_end = time.time()
            
            logger.info(f"배치 삽입 완료: {len(batch)}개 항목, 소요시간: {batch_insert_end - batch_insert_start:.4f}초")
        except Exception as e:
            logger.error(f"배치 삽입 중 오류 발생: {str(e)}")
            # 오류 발생 시 개별 항목 삽입 시도
            for item in batch:
                try:
                    interact_manager.batch_insert_data(domain, [item])
                    logger.info(f"개별 항목 삽입 성공: {item.get('passage_uid', 'unknown')}")
                except Exception as single_error:
                    logger.error(f"개별 항목 삽입 실패: {item.get('passage_uid', 'unknown')}, 오류={str(single_error)}")
    
    try:
        request_data = request.json
        if not request_data:
            insert_logger.error("요청 본문이 비어있습니다.")
            return jsonify({
                "result_code": "F000001",
                "message": "요청 본문이 비어있습니다."
            }), 400

        if "documents" not in request_data:
            insert_logger.error("documents 필드는 필수입니다.")
            return jsonify({
                "result_code": "F000002",
                "message": "documents 필드는 필수입니다."
            }), 400

        if not isinstance(request_data["documents"], list) or len(request_data["documents"]) == 0:
            insert_logger.error("documents는 최소 1개 이상의 문서를 포함해야 합니다.")
            return jsonify({
                "result_code": "F000003",
                "message": "documents는 최소 1개 이상의 문서를 포함해야 합니다."
            }), 400

        # ignore 옵션 처리 (기본값: True)
        ignore = request_data.get('ignore', True)
        insert_logger.info(f"중복 문서 처리 모드: ignore={ignore}")

        # 필수 필드 검증
        required_fields = ['domain', 'title', 'author', 'text', 'tags']
        
        # 요청 유효성 검사 결과 저장용 변수
        validation_results = []
        valid_documents = []
        
        # 유효성 검사 단계
        validation_start = time.time()
        for doc_index, doc in enumerate(request_data["documents"]):
            # 필수 필드 검증
            missing_fields = [field for field in required_fields if field not in doc]
            if missing_fields:
                validation_results.append({
                    "status": "error",
                    "result_code": "F000004",
                    "message": f"필수 필드가 누락되었습니다: {', '.join(missing_fields)}",
                    "title": doc.get('title', 'unknown'),
                    "index": doc_index
                })
                continue

            if 'date' not in doc['tags']:
                validation_results.append({
                    "status": "error",
                    "result_code": "F000005",
                    "message": "tags.date는 필수 입력값입니다.",
                    "title": doc['title'],
                    "index": doc_index
                })
                continue
                
            # 유효한 문서 목록에 추가
            doc['_index'] = doc_index  # 원래 인덱스 추적용
            valid_documents.append(doc)
            
        validation_end = time.time()
        logger.info(f"[TIMING] 문서 유효성 검사 완료: {len(valid_documents)}/{len(request_data['documents'])} 유효, 소요시간: {validation_end - validation_start:.4f}초")
        insert_logger.info(f"문서 유효성 검사 완료: {len(valid_documents)}/{len(request_data['documents'])} 유효")
        
        # 상태별 카운터 초기화
        status_counts = {
            "success": 0,  # 새로 삽입됨
            "skipped": 0,  # 중복으로 무시됨
            "error": 0     # 오류 발생
        }
        
        # 초기 에러 결과 집계
        for result in validation_results:
            status_counts["error"] += 1
        
        # 처리 결과 저장 배열
        results = validation_results.copy()  # 유효성 검사 결과로 초기화
        
        # 도메인별로 문서 그룹화 (배치 처리를 위해)
        domain_documents = {}
        for doc in valid_documents:
            domain = doc['domain']
            if domain not in domain_documents:
                domain_documents[domain] = []
            domain_documents[domain].append(doc)
        
        insert_logger.info(f"도메인별 문서 분류: {', '.join([f'{domain}({len(docs)}개)' for domain, docs in domain_documents.items()])}")
        
        # 각 도메인별 처리
        for domain, docs in domain_documents.items():
            domain_start = time.time()
            logger.info(f"[TIMING] 도메인 '{domain}' 처리 시작: {len(docs)}개 문서")
            insert_logger.info(f"도메인 '{domain}' 처리 시작: {len(docs)}개 문서")
            
            try:
                # 도메인 생성 (없는 경우에만)
                if domain not in milvus_db.get_list_collection():
                    logger.info(f"Creating new domain: {domain}")
                    insert_logger.info(f"새 도메인 생성: {domain}")
                    interact_manager.create_domain(domain)
                    # 새로 생성된 컬렉션 로드
                    collection = Collection(domain)
                    collection.load()
                    logger.info(f"New collection {domain} created and loaded")
                
                # 1. 문서 ID 해시 계산 및 중복 확인 준비
                doc_hashes = []
                doc_hash_map = {}  # 해시 ID -> 원본 문서 매핑
                
                for doc in docs:
                    # doc_id 생성 및 해시 처리
                    raw_doc_id = f"{doc['tags']['date'].replace('-','')}-{doc['title']}-{doc['author']}"
                    hashed_doc_id = env_manager.data_p.hash_text(raw_doc_id, hash_type='blake')
                    
                    # 추적을 위해 문서에 해시 ID 저장
                    doc['_hashed_doc_id'] = hashed_doc_id
                    doc['_raw_doc_id'] = raw_doc_id
                    
                    doc_hashes.append(hashed_doc_id)
                    doc_hash_map[hashed_doc_id] = doc
                
                insert_logger.info(f"문서 ID 해시 계산 완료: {len(doc_hashes)}개")
                
                # 컬렉션 로드
                collection = Collection(domain)
                collection.load()
                
                # 2. 중복 문서 일괄 확인 (단일 쿼리로 모든 문서 체크)
                duplicate_check_start = time.time()
                
                # 개선된 방식으로 중복 체크 - pipe.py의 check_duplicates 함수 사용
                logger.info(f"[DUPLICATION_CHECK] 시작: 총 {len(doc_hashes)}개 문서 ID 중복 검사 (도메인: {domain}, ignore={ignore})")
                insert_logger.info(f"중복 검사 시작: 총 {len(doc_hashes)}개 문서 ID (도메인: {domain}, ignore={ignore})")
                
                # 전체 doc_id 목록 로깅 (50개 미만일 경우)
                if len(doc_hashes) <= 50:
                    logger.info(f"[DUPLICATION_CHECK] 검사할 모든 doc_id 목록: {doc_hashes}")
                    insert_logger.info(f"검사할 모든 doc_id 목록: {doc_hashes}")
                else:
                    sample_ids = doc_hashes[:5] + ['...'] + doc_hashes[-5:]
                    logger.info(f"[DUPLICATION_CHECK] 검사할 doc_id 샘플: {sample_ids}")
                    insert_logger.info(f"검사할 doc_id 샘플: {sample_ids[:5]} ... {sample_ids[-5:]}")
                
                if doc_hashes:
                    try:
                        # doc_id 해시 재확인 - 모든 ID가 올바른 형식인지 검사
                        valid_doc_hashes = []
                        for doc_id in doc_hashes:
                            # 올바른 해시 형식 확인 (최소 길이와 16진수 문자만 포함)
                            if doc_id and len(doc_id) >= 32 and all(c in '0123456789abcdef' for c in doc_id.lower()):
                                valid_doc_hashes.append(doc_id)
                            else:
                                logger.warning(f"[DUPLICATION_CHECK] 잘못된 doc_id 형식: {doc_id}")
                                insert_logger.warning(f"잘못된 doc_id 형식: {doc_id}")
                        
                        if len(valid_doc_hashes) < len(doc_hashes):
                            logger.warning(f"[DUPLICATION_CHECK] 일부 doc_id가 필터링됨: 총 {len(doc_hashes)}개 중 {len(valid_doc_hashes)}개만 유효함")
                            insert_logger.warning(f"일부 doc_id가 필터링됨: 총 {len(doc_hashes)}개 중 {len(valid_doc_hashes)}개만 유효함")
                        
                        # interact_manager의 check_duplicates 함수 사용
                        existing_doc_ids = interact_manager.check_duplicates(valid_doc_hashes, domain)
                        
                        # 디버깅: 중복 검사 결과 상세 로깅
                        insert_logger.info(f"중복 검사 결과 - 타입: {type(existing_doc_ids)}, 값: {existing_doc_ids}")
                        
                        logger.info(f"[DUPLICATION_CHECK] 완료: 총 {len(valid_doc_hashes)}개 문서 중 {len(existing_doc_ids)}개 중복 발견")
                        insert_logger.info(f"중복 검사 완료: 총 {len(valid_doc_hashes)}개 문서 중 {len(existing_doc_ids)}개 중복 발견")
                        
                        # 중복 ID가 있다면 로그에 명확하게 표시
                        if existing_doc_ids:
                            if len(existing_doc_ids) <= 50:
                                logger.info(f"[DUPLICATION_CHECK] 중복 문서 ID 전체 목록: {existing_doc_ids}")
                                insert_logger.info(f"중복 문서 ID 전체 목록: {existing_doc_ids}")
                            else:
                                display_dupes = existing_doc_ids[:20]
                                more_count = len(existing_doc_ids) - len(display_dupes)
                                logger.info(f"[DUPLICATION_CHECK] 중복 문서 ID 일부: {display_dupes} 외 {more_count}개")
                                insert_logger.info(f"중복 문서 ID 일부: {display_dupes} 외 {more_count}개")
                            logger.info(f"[DUPLICATION_CHECK] 처리 방식: {'건너뛰기 (ignore=true)' if ignore else '삭제 후 재삽입 (ignore=false)'}")
                            insert_logger.info(f"중복 처리 방식: {'건너뛰기 (ignore=true)' if ignore else '삭제 후 재삽입 (ignore=false)'}")
                        else:
                            logger.info(f"[DUPLICATION_CHECK] 중복 문서 없음")
                            insert_logger.info(f"중복 문서 없음")
                    except Exception as e:
                        logger.error(f"[DUPLICATION_CHECK] 심각한 오류: 중복 체크 실패: {str(e)}")
                        insert_logger.error(f"중복 체크 심각한 오류: {str(e)}")
                        import traceback
                        logger.error(f"[DUPLICATION_CHECK] 상세 오류: {traceback.format_exc()}")
                        insert_logger.error(f"상세 오류: {traceback.format_exc()}")
                        existing_doc_ids = []
                else:
                    existing_doc_ids = []
                    logger.warning(f"[DUPLICATION_CHECK] 경고: 검사할 문서가 없습니다 (도메인: {domain})")
                    insert_logger.warning(f"경고: 검사할 문서가 없습니다 (도메인: {domain})")
                
                duplicate_check_end = time.time()
                logger.info(f"[TIMING] 중복 문서 체크 완료: {len(existing_doc_ids)}/{len(doc_hashes)}개 중복, 소요시간: {duplicate_check_end - duplicate_check_start:.4f}초")
                
                # 3. 문서 분류 (삭제할 문서, 건너뛸 문서, 삽입할 문서)
                docs_to_delete = []
                docs_to_skip = []
                docs_to_insert = []
                
                for doc_id in doc_hashes:
                    doc = doc_hash_map[doc_id]
                    # 디버깅: 각 문서의 중복 여부 확인 로깅
                    is_duplicate = doc_id in existing_doc_ids
                    insert_logger.info(f"문서 중복 여부 확인: ID={doc_id}, 중복={is_duplicate}")
                    
                    if is_duplicate:
                        if ignore:
                            # 중복이고 ignore=true면 건너뜀
                            docs_to_skip.append(doc)
                            logger.info(f"[DUPLICATION_CHECK] 문서 건너뛰기: {doc_id} (ignore=true)")
                            insert_logger.info(f"문서 건너뛰기: {doc_id} (ignore=true)")
                        else:
                            # 중복이지만 ignore=false면 삭제 후 재삽입
                            docs_to_delete.append(doc)
                            docs_to_insert.append(doc)
                            logger.info(f"[DUPLICATION_CHECK] 문서 재삽입: {doc_id} (ignore=false)")
                            insert_logger.info(f"문서 재삽입: {doc_id} (ignore=false)")
                    else:
                        # 중복이 아니면 삽입
                        docs_to_insert.append(doc)
                        insert_logger.info(f"새 문서 삽입: {doc_id}")
                
                # 주요 동작 결과 요약 로그
                logger.info(f"[DUPLICATION_CHECK] 결과 요약: 총 {len(doc_hashes)}개 문서 중 {len(docs_to_skip)}개 건너뛰기, {len(docs_to_delete)}개 재삽입, {len(docs_to_insert) - len(docs_to_delete)}개 신규 삽입")
                insert_logger.info(f"결과 요약: 총 {len(doc_hashes)}개 문서 중 {len(docs_to_skip)}개 건너뛰기, {len(docs_to_delete)}개 재삽입, {len(docs_to_insert) - len(docs_to_delete)}개 신규 삽입")
                
                # 4. 중복 문서 일괄 삭제 (ignore=false인 경우)
                if not ignore and docs_to_delete:
                    delete_start = time.time()
                    delete_ids = [doc['_hashed_doc_id'] for doc in docs_to_delete]
                    insert_logger.info(f"삭제할 문서 ID 목록: {delete_ids[:5]}{'...' if len(delete_ids) > 5 else ''}")
                    
                    # 배치 단위로 삭제 (쿼리 크기 제한 고려)
                    delete_batch_size = 500
                    total_deleted = 0
                    
                    for i in range(0, len(delete_ids), delete_batch_size):
                        batch_ids = delete_ids[i:i+delete_batch_size]
                        ids_list = ", ".join([f'"{id}"' for id in batch_ids])  # 각 ID를 큰따옴표로 묶고 쉼표로 구분
                        expr = f'doc_id in [{ids_list}]'  # 올바른 IN 연산자 형식 사용: doc_id in ["id1", "id2", ...]
                        
                        try:
                            deleted_result = collection.delete(expr)
                            # MutationResult 객체에서 삭제된 항목 수 추출
                            if hasattr(deleted_result, 'delete_count'):
                                deleted_count = deleted_result.delete_count
                            else:
                                # 객체 자체가 정수일 경우 (이전 버전 호환) 또는 속성명이 다를 경우 대비
                                try:
                                    deleted_count = int(deleted_result)
                                except (TypeError, ValueError):
                                    # 다른 가능한 속성 이름 시도
                                    deleted_count = getattr(deleted_result, 'num_deleted', 0) or getattr(deleted_result, 'count', 0)
                                    
                            total_deleted += deleted_count
                            logger.info(f"[TIMING] 배치 삭제 완료 ({i//delete_batch_size + 1}/{(len(delete_ids)+delete_batch_size-1)//delete_batch_size}): {deleted_count}개 항목")
                            insert_logger.info(f"배치 삭제 완료 ({i//delete_batch_size + 1}/{(len(delete_ids)+delete_batch_size-1)//delete_batch_size}): {deleted_count}개 항목")
                        except Exception as e:
                            logger.error(f"배치 삭제 오류 (배치 {i//delete_batch_size + 1}): {str(e)}")
                            insert_logger.error(f"배치 삭제 오류 (배치 {i//delete_batch_size + 1}): {str(e)}")
                    
                    # 삭제 후 즉시 flush하여 변경사항 적용
                    collection.flush()
                    insert_logger.info(f"삭제 후 flush 완료")
                    
                    delete_end = time.time()
                    logger.info(f"[TIMING] 문서 일괄 삭제 완료: {total_deleted}개 항목, 소요시간: {delete_end - delete_start:.4f}초")
                
                # 5. 삽입할 문서 처리 (청킹 및 임베딩)
                if docs_to_insert:
                    insert_start = time.time()
                    
                    # 임베딩 스레드 풀 생성
                    max_embed_threads = min(
                        int(os.getenv('INSERT_CHUNK_THREADS', '10')),  # 기본값: 10
                        10  # 최대 10개까지 제한
                    )
                    
                    # 문서 병렬 처리를 위한 스레드 풀 크기 설정
                    max_document_threads = min(
                        int(os.getenv('INSERT_DOCUMENT_THREADS', '5')),  # 기본값: 5
                        len(docs_to_insert),  # 문서 수보다 많은 스레드는 불필요
                        5  # 하드 리밋 (선택적)
                    )
                    logger.info(f"[TIMING] 문서 병렬 처리 시작: 총 {len(docs_to_insert)}개 문서, 최대 {max_document_threads}개 스레드")
                    
                    # 모든 문서의 모든 청크를 담을 리스트 (스레드 안전 처리 필요)
                    all_chunks = []
                    chunks_lock = threading.Lock()  # 스레드 안전성을 위한 락
                    chunk_to_doc_map = {}  # 청크 -> 원본 문서 매핑
                    
                    # 글로벌 배치 워커 시작
                    from src.pipe import InteractManager
                    InteractManager.start_batch_worker()
                    logger.info("글로벌 배치 처리 워커 시작됨")
                    
                    # 배치 처리 설정
                    batch_size = int(os.getenv('BATCH_SIZE', '300'))  # 기본값: 300
                    logger.info(f"배치 크기 설정: {batch_size}")
                    InteractManager.batch_size = batch_size  # 배치 크기 설정

                    # 문서 처리 함수 정의
                    def process_document(doc):
                        try:
                            doc_process_start = time.time()
                            doc_title = doc.get('title', 'unknown')
                            
                            # 문서 ID 확인 및 설정
                            if '_hashed_doc_id' in doc:
                                doc_hashed_id = doc['_hashed_doc_id']
                            else:
                                doc_hashed_id = doc.get('id', 'unknown')
                                
                            thread_id = threading.get_ident()  # 현재 스레드 ID 가져오기
                            
                            # 문서 처리 시작 로그 개선
                            insert_logger.info(f"문서 처리 시작: {doc_title} (ID: {doc_hashed_id}, Thread: {thread_id})")
                            logger.info(f"[TIMING] 문서 처리 시작: {doc_title} (ID: {doc_hashed_id})")
                            
                            # 문서 텍스트 추출
                            if isinstance(doc, dict) and 'text' in doc:
                                doc_text = doc['text']
                            else:
                                # 문서 자체가 텍스트인 경우
                                doc_text = doc
                            
                            # 1. 청킹 단계 시간 측정
                            chunking_start = time.time()
                            # 문서 청킹
                            chunks = interact_manager.chunk_document(doc_text)
                            chunking_end = time.time()
                            chunking_duration = chunking_end - chunking_start
                            
                            logger.info(f"[TIMING] 문서 '{doc_title}' 청킹 완료: {len(chunks)}개 청크, 소요시간: {chunking_duration:.4f}초")
                            insert_logger.info(f"문서 '{doc_title}' 청킹 완료: {len(chunks)}개 청크, 소요시간: {chunking_duration:.4f}초 (Thread: {thread_id})")
                            
                            # 청크가 없으면 종료
                            if not chunks:
                                logger.warning(f"문서 '{doc_title}'에서 생성된 청크가 없습니다.")
                                insert_logger.warning(f"문서 '{doc_title}'에서 생성된 청크가 없습니다. (Thread: {thread_id})")
                                doc_process_end = time.time()
                                doc_process_duration = doc_process_end - doc_process_start
                                logger.info(f"[TIMING] 문서 처리 완료(청크 없음): {doc_title}, 소요시간: {doc_process_duration:.4f}초")
                                insert_logger.info(f"문서 처리 완료(청크 없음): {doc_title}, 소요시간: {doc_process_duration:.4f}초 (Thread: {thread_id})")
                                return 0
                            
                            insert_logger.info(f"문서 '{doc_title}'의 청크 처리 시작: 총 {len(chunks)}개 청크, Thread: {thread_id}")
                            
                            # 2. 청크 임베딩 처리 단계 시간 측정
                            embedding_start = time.time()
                            
                            # 청크 임베딩 병렬 처리 스레드 수 설정
                            max_chunk_threads = min(
                                int(os.getenv('INSERT_CHUNK_THREADS', '10')),  # 기본값: 10
                                len(chunks),  # 청크 수보다 많은 스레드는 불필요
                                10  # 최대 10개로 제한
                            )
                            
                            # 청크 중복 처리 방지를 위한 집합 초기화
                            processed_chunk_ids = set()
                            
                            # 각 청크 처리에 필요한 도메인 정보 준비
                            domain = doc.get('domain', 'general')
                            
                            # 문서 ID 미리 저장 (문서의 여러 위치에서 일관되게 사용하기 위함)
                            # 이미 위에서 설정한 doc_hashed_id 재사용
                            doc_title = doc.get('title', '제목 없음')
                            
                            # raw_doc_id 준비
                            raw_doc_id = doc.get('_raw_doc_id', doc_hashed_id)
                            
                            # 임베딩 생성 함수 정의
                            def process_chunk(chunk, index):
                                """
                                청크를 처리하고 임베딩을 추가합니다.
                                
                                Args:
                                    chunk (dict 또는 tuple): 청크 데이터
                                    index (int): 청크 인덱스
                                    
                                Returns:
                                    dict: 처리된 청크 데이터 또는 None (오류 시)
                                """
                                try:
                                    # 튜플 형태의 청크를 사전 형태로 변환
                                    if isinstance(chunk, tuple):
                                        # 튜플을 딕셔너리로 변환
                                        chunk_dict = {'text': str(chunk[0]) if len(chunk) > 0 else ''}
                                        # 추가 정보가 있는 경우 처리
                                        if len(chunk) > 1:
                                            chunk_dict['metadata'] = chunk[1] if isinstance(chunk[1], dict) else {}
                                        chunk = chunk_dict
                                        logger.info(f"튜플 형태의 청크를 딕셔너리로 변환: 청크 #{index}")
                                    
                                    # 필수 필드 검증
                                    if not chunk.get('text', '').strip():
                                        logger.warning(f"빈 텍스트 청크 건너뜀: {doc.get('doc_id', 'unknown')}, 청크 #{index}")
                                        return None
                                    
                                    # 메타데이터 및 필수 필드 설정
                                    if 'metadata' not in chunk:
                                        chunk['metadata'] = {}
                                    
                                    # 문서 메타데이터 상속
                                    chunk['metadata'].update(doc.get('metadata', {}))
                                    
                                    # 문서 ID와 청크 번호 설정
                                    chunk['doc_id'] = doc_hashed_id  # 해시된 문서 ID 사용
                                    chunk['raw_doc_id'] = raw_doc_id  # 원본 문서 ID 보존
                                    chunk['passage_id'] = index
                                    
                                    # 문서 메타데이터 설정
                                    chunk['title'] = doc.get('title', '')
                                    chunk['author'] = doc.get('author', '')
                                    chunk['domain'] = domain
                                    
                                    # 원본 문서 정보 추가 (명시적으로 청크가 어떤 문서에서 왔는지 표시)
                                    chunk['source_doc'] = {
                                        'id': doc_hashed_id,
                                        'raw_id': raw_doc_id,
                                        'title': doc.get('title', '')
                                    }
                                    
                                    # 문서 태그 및 정보 설정
                                    chunk['info'] = doc.get('info', {})
                                    chunk['tags'] = doc.get('tags', {})
                                    
                                    # 메모리 사용량 추적
                                    memory_usage = get_memory_usage()
                                    logger.info(f"[MEMORY] 청크 처리 전 메모리 사용량: {memory_usage:.2f} MB")
                                    
                                    # 임베딩 추가
                                    processed_chunk = interact_manager.embed_and_prepare_chunk(chunk)
                                    
                                    if processed_chunk is None:
                                        logger.warning(f"청크 처리 실패: {doc_hashed_id}, 청크 #{index}")
                                        return None
                                    
                                    # 해시 기반 고유 ID 생성 (passage_uid)
                                    if 'passage_uid' not in processed_chunk:
                                        import hashlib
                                        text_hash = hashlib.sha512(processed_chunk['text'].encode('utf-8')).hexdigest()
                                        passage_id = str(processed_chunk.get('passage_id', '0'))
                                        # 문서 ID를 포함하여 문서 간 고유성 보장
                                        processed_chunk['passage_uid'] = f"{doc_hashed_id}_{text_hash}_{passage_id}"
                                    
                                    # 처리된 청크 데이터를 글로벌 배치 큐에 직접 추가
                                    from src.pipe import InteractManager
                                    added_to_batch = InteractManager.add_to_global_batch(processed_chunk, domain)
                                    
                                    if added_to_batch:
                                        logger.info(f"청크 글로벌 배치 큐에 추가 완료: {doc_hashed_id}, 청크 #{index}")
                                    
                                    # 처리된 청크 반환 (배치 처리를 위해)
                                    return processed_chunk
                                    
                                except Exception as e:
                                    logger.error(f"청크 처리 중 오류: {str(e)}, 청크 #{index}")
                                    import traceback
                                    logger.error(f"오류 세부 정보: {traceback.format_exc()}")
                                    return None
                            
                            # 청크 병렬 처리
                            with concurrent.futures.ThreadPoolExecutor(max_workers=max_chunk_threads) as executor:
                                future_to_chunk = {executor.submit(process_chunk, chunk, i): i for i, chunk in enumerate(chunks)}
                                doc_prepared_chunks = []
                                completed_chunks = []  # 완료된 청크 추적
                                
                                # 청크 처리 결과 수집
                                for future in concurrent.futures.as_completed(future_to_chunk):
                                    chunk_index = future_to_chunk[future]
                                    try:
                                        data = future.result()
                                        if data:
                                            # 중복 처리 방지 - doc_prepared_chunks에는 한 번만 추가 (통계용)
                                            doc_prepared_chunks.append(data)
                                            completed_chunks.append(chunk_index)
                                    except Exception as e:
                                        logger.error(f"문서 '{doc_title}' 청크 {chunk_index} 처리 실패: {str(e)}")
                                        # 실패 카운터 증가
                                        if not hasattr(process_document, 'failed_count'):
                                            process_document.failed_count = 0
                                        process_document.failed_count += 1
                                        
                                        # 일정 개수마다 통계 로깅
                                        if process_document.failed_count % 5 == 0:
                                            logger.warning(f"총 {process_document.failed_count}개 청크 처리 실패")
                            
                            embedding_end = time.time()
                            embedding_duration = embedding_end - embedding_start
                            
                            # 청크 처리 완료 로그 간소화
                            logger.info(f"[TIMING] 문서 '{doc_title}'의 청크 처리 완료: 성공={len(doc_prepared_chunks)}, 실패={len(chunks)-len(doc_prepared_chunks)}, 총={len(chunks)}개 청크, 소요시간: {embedding_duration:.4f}초")
                            
                            # 전체 처리 완료 시간 측정
                            doc_process_end = time.time()
                            doc_process_duration = doc_process_end - doc_process_start
                            
                            # 단계별 시간 비율 계산
                            total_time = max(0.0001, doc_process_duration)  # 0으로 나누기 방지
                            chunking_percent = chunking_duration / total_time * 100
                            embedding_percent = embedding_duration / total_time * 100
                            
                            # 단계별 처리 시간 종합 - 핵심 정보이므로 유지
                            insert_logger.info(f"문서 처리 완료: {doc_title}, 청크 수: {len(chunks)}, 성공 임베딩: {len(doc_prepared_chunks)}, 소요시간: {doc_process_duration:.4f}초 (Thread: {thread_id})")
                            insert_logger.info(f"문서 처리 단계별 시간: 청킹={chunking_duration:.4f}초({chunking_percent:.1f}%), 임베딩={embedding_duration:.4f}초({embedding_percent:.1f}%) (Thread: {thread_id})")
                            
                            # 성공적으로 처리된 청크 수 반환
                            return len(doc_prepared_chunks)
                        except Exception as e:
                            logger.error(f"문서 처리 오류 (title: {doc.get('title', 'unknown')}): {str(e)}")
                            insert_logger.error(f"문서 처리 오류 (title: {doc.get('title', 'unknown')}): {str(e)}")
                            return 0
                    
                    # 문서 병렬 처리 실행
                    chunking_start = time.time()
                    
                    # 전체 GPU 세마포어 상태 확인
                    if hasattr(interact_manager.emb_model, 'get_gpu_semaphore'):
                        try:
                            from src.pipe import InteractManager
                            # 새로운 메서드 사용
                            max_workers = InteractManager.get_max_workers()
                            active_workers = InteractManager.get_active_workers()
                            sem_value = InteractManager.get_gpu_semaphore_value()
                            
                            insert_logger.info(f"문서 처리 완료 후 GPU 세마포어 상태: 활성작업={active_workers}/{max_workers}, 세마포어값={sem_value}")
                        except Exception as sem_err:
                            insert_logger.warning(f"GPU 세마포어 상태 확인 실패: {str(sem_err)}")
                    
                    # 총 처리 결과 초기화
                    total_chunks = 0
                    
                    # 문서 처리를 병렬로 수행
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_document_threads) as doc_executor:
                            # 각 문서에 대한 처리 작업 제출
                            future_results = list(doc_executor.map(process_document, docs_to_insert))
                            # 총 성공 청크 수 계산
                            total_chunks = sum(future_results)
                    except Exception as exec_error:
                        logger.error(f"문서 병렬 처리 중 오류 발생: {str(exec_error)}")
                        insert_logger.error(f"문서 병렬 처리 중 오류 발생: {str(exec_error)}")
                        return jsonify({"error": f"문서 처리 오류: {str(exec_error)}"}), 500
                    
                    # 전체 GPU 세마포어 상태 확인
                    if hasattr(interact_manager.emb_model, 'get_gpu_semaphore'):
                        try:
                            from src.pipe import InteractManager
                            # 새로운 메서드 사용
                            max_workers = InteractManager.get_max_workers()
                            active_workers = InteractManager.get_active_workers()
                            sem_value = InteractManager.get_gpu_semaphore_value()
                            
                            insert_logger.info(f"문서 처리 완료 후 GPU 세마포어 상태: 활성작업={active_workers}/{max_workers}, 세마포어값={sem_value}")
                        except Exception as sem_err:
                            insert_logger.warning(f"GPU 세마포어 상태 확인 실패: {str(sem_err)}")
                    
                    chunking_end = time.time()
                    logger.info(f"[TIMING] 문서 처리 완료: {len(docs_to_insert)}개 문서 -> {total_chunks}개 청크 성공, 소요시간: {chunking_end - chunking_start:.4f}초")
                    insert_logger.info(f"문서 처리 완료: {len(docs_to_insert)}개 문서 -> {total_chunks}개 청크 성공, 소요시간: {chunking_end - chunking_start:.4f}초")
                    
                    insert_end = time.time()
                    logger.info(f"[TIMING] 문서 삽입 처리 완료: 소요시간: {insert_end - insert_start:.4f}초")
                    insert_logger.info(f"문서 삽입 처리 완료: 소요시간: {insert_end - insert_start:.4f}초")
                    
                # 6. 결과 구성
                for doc in docs_to_skip:
                    results.append({
                        "status": "skipped",
                        "result_code": "F000000",
                        "message": "이미 존재하는 문서로 건너뛰었습니다.",
                        "doc_id": doc['_hashed_doc_id'],
                        "raw_doc_id": doc['_raw_doc_id'],
                        "domain": doc['domain'],
                        "title": doc['title'],
                        "index": doc['_index']
                    })
                    status_counts["skipped"] += 1
                
                for doc in docs_to_insert:
                    if doc in docs_to_delete:
                        # 삭제 후 재삽입된 문서
                        results.append({
                            "status": "updated",
                            "result_code": "F000000",
                            "message": "기존 문서가 삭제되고 새 문서로 대체되었습니다.",
                            "doc_id": doc['_hashed_doc_id'],
                            "raw_doc_id": doc['_raw_doc_id'],
                            "domain": doc['domain'],
                            "title": doc['title'],
                            "index": doc['_index']
                        })
                    else:
                        # 새로 삽입된 문서
                        results.append({
                            "status": "success",
                            "result_code": "F000000",
                            "message": "문서가 성공적으로 저장되었습니다.",
                            "doc_id": doc['_hashed_doc_id'],
                            "raw_doc_id": doc['_raw_doc_id'],
                            "domain": doc['domain'],
                            "title": doc['title'],
                            "index": doc['_index']
                        })
                    status_counts["success"] += 1
                
                domain_end = time.time()
                logger.info(f"[TIMING] 도메인 '{domain}' 처리 완료: 소요시간: {domain_end - domain_start:.4f}초")
                
            except Exception as domain_error:
                logger.error(f"도메인 '{domain}' 처리 중 오류: {str(domain_error)}")
                # 도메인 처리 실패 시 해당 도메인의 모든 문서를 오류로 처리
                for doc in docs:
                    results.append({
                        "status": "error",
                        "result_code": "F000006",
                        "message": f"도메인 처리 중 오류 발생: {str(domain_error)}",
                        "title": doc['title'],
                        "index": doc['_index']
                    })
                    status_counts["error"] += 1
        
        # 결과 정렬 (원래 인덱스 기준)
        results.sort(key=lambda x: x.get("index", 0))
        
        # 인덱스 필드 제거 (클라이언트에게 불필요)
        for result in results:
            if "index" in result:
                del result["index"]
        
        # 전체 상태 결정
        if status_counts["error"] == len(request_data["documents"]):
            overall_status = "error"  # 모두 실패
        elif status_counts["error"] == 0 and status_counts["skipped"] == 0:
            overall_status = "success"  # 모두 성공
        elif status_counts["error"] == 0:
            overall_status = "partial_success"  # 일부는 성공하고 일부는 건너뜀
        else:
            overall_status = "partial_error"  # 일부 실패
        
        # API 전체 실행 시간 측정 완료
        api_end_time = time.time()
        api_duration = api_end_time - api_start_time
        logger.info(f"=== INSERT API END === Duration: {api_duration:.4f}s, Status: {overall_status}, Documents: {len(request_data['documents'])}, Success: {status_counts['success']}, Skipped: {status_counts['skipped']}, Error: {status_counts['error']}")
        
        return jsonify({
            "status": overall_status,
            "message": f"총 {len(request_data['documents'])}개 문서 중 {status_counts['success']}개 성공, {status_counts['skipped']}개 건너뜀, {status_counts['error']}개 실패",
            "status_counts": status_counts,
            "results": results,
            "performance": {
                "total_time": api_duration
            }
        }), 200 if overall_status != "error" else 500
        
    except Exception as e:
        # API 에러 시간 측정
        api_end_time = time.time()
        api_duration = api_end_time - api_start_time if 'api_start_time' in locals() else 0
        logger.error(f"=== INSERT API ERROR === Duration: {api_duration:.4f}s, Error: {str(e)}")
        
        return jsonify({
            "status": "error",
            "result_code": "F000007",
            "message": f"요청 처리 중 오류가 발생했습니다: {str(e)}"
        }), 500

@app.route('/rag/insert/raw', methods=['POST'])
def insert_raw_data():
    '''
    텍스트를 분할하지 않고 그대로 저장하는 API
    {
        "documents": [
            {
                "doc_id": "unique_document_id",  # 필수: 사용자가 지정한 고유 문서 ID
                "passage_id": 1,  # 필수: 사용자가 지정한 passage ID
                "domain": "news",
                "title": "메타버스 뉴스",
                "author": "삼성전자",
                "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다...",
                "info": {
                    "press_num": "비즈니스 워치",
                    "url": "http://example.com/news/1"
                },
                "tags": {
                    "date": "20240315",
                    "user": "admin"
                }
            }
        ],
        "ignore": true  # true: 중복 시 건너뜀, false: 중복 시 삭제 후 재생성
    }
    '''
    try:
        request_data = request.json
        if not request_data:
            return jsonify({
                "result_code": "F000001",
                "message": "요청 본문이 비어있습니다."
            }), 400

        if "documents" not in request_data:
            return jsonify({
                "result_code": "F000002",
                "message": "documents 필드는 필수입니다."
            }), 400

        if not isinstance(request_data["documents"], list) or len(request_data["documents"]) == 0:
            return jsonify({
                "result_code": "F000003",
                "message": "documents는 최소 1개 이상의 문서를 포함해야 합니다."
            }), 400

        # ignore 옵션 처리 (기본값: True)
        ignore = request_data.get('ignore', True)

        # 필수 필드 검증 (doc_id, passage_id 추가)
        required_fields = ['doc_id', 'passage_id', 'domain', 'title', 'author', 'text', 'tags']
        results = []
        
        # 상태별 카운터 초기화
        status_counts = {
            "success": 0,  # 새로 삽입됨
            "updated": 0,  # 업데이트됨
            "skipped": 0,  # 중복으로 무시됨
            "error": 0     # 오류 발생
        }
        
        # 유효성 검사 단계 - 유효하지 않은 문서 필터링
        valid_documents = []
        for doc in request_data["documents"]:
            try:
                # 필수 필드 검증
                missing_fields = [field for field in required_fields if field not in doc]
                if missing_fields:
                    results.append({
                        "status": "error",
                        "result_code": "F000004",
                        "message": f"필수 필드가 누락되었습니다: {', '.join(missing_fields)}",
                        "title": doc.get('title', 'unknown')
                    })
                    status_counts["error"] += 1
                    continue

                if 'date' not in doc['tags']:
                    results.append({
                        "status": "error",
                        "result_code": "F000005",
                        "message": "tags.date는 필수 입력값입니다.",
                        "title": doc['title']
                    })
                    status_counts["error"] += 1
                    continue

                # passage_id가 정수인지 확인
                try:
                    passage_id = int(doc['passage_id'])
                    doc['passage_id'] = passage_id
                except (ValueError, TypeError):
                    results.append({
                        "status": "error",
                        "result_code": "F000008",
                        "message": "passage_id는 정수여야 합니다.",
                        "title": doc['title']
                    })
                    status_counts["error"] += 1
                    continue
                
                # 유효한 문서 목록에 추가
                valid_documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error validating document: {str(e)}")
                results.append({
                    "status": "error",
                    "result_code": "F000006",
                    "message": f"문서 유효성 검사 중 오류가 발생했습니다: {str(e)}",
                    "title": doc.get('title', 'unknown')
                })
                status_counts["error"] += 1
        
        # 도메인별로 문서 그룹화
        domain_documents = {}
        for doc in valid_documents:
            domain = doc['domain']
            if domain not in domain_documents:
                domain_documents[domain] = []
            domain_documents[domain].append(doc)
        
        # 각 도메인별 처리
        for domain, docs in domain_documents.items():
            try:
                # 도메인이 없으면 생성
                if domain not in milvus_db.get_list_collection():
                    interact_manager.create_domain(domain)
                    # 새로 생성된 컬렉션 로드
                    collection = Collection(domain)
                    collection.load()
                    print(f"[DEBUG] New collection {domain} created and loaded")
                
                # 문서 처리 함수 정의
                def process_document(doc):
                    try:
                        # 데이터 삽입 시도 (raw_insert_data 메소드 사용)
                        insert_status = interact_manager.raw_insert_data(
                            doc['domain'], 
                            doc['doc_id'],  # 사용자가 제공한 doc_id 사용
                            doc['passage_id'],  # 사용자가 제공한 passage_id 사용
                            doc['title'], 
                            doc['author'], 
                            doc['text'], 
                            doc.get('info', {}),
                            doc['tags'],
                            ignore=ignore  # 전체 요청에 대한 ignore 값 사용
                        )
                        
                        result = {
                            "doc_id": doc['doc_id'],
                            "passage_id": doc['passage_id'],
                            "domain": doc['domain'],
                            "title": doc['title']
                        }
                        
                        if insert_status == "skipped":
                            result.update({
                                "status": "skipped",
                                "result_code": "F000000",
                                "message": "이미 존재하는 문서로 건너뛰었습니다."
                            })
                        elif insert_status == "updated":
                            result.update({
                                "status": "updated",
                                "result_code": "F000000",
                                "message": "기존 문서를 삭제하고 새로운 문서로 업데이트했습니다."
                            })
                        else:  # success
                            result.update({
                                "status": "success",
                                "result_code": "F000000",
                                "message": "문서가 성공적으로 저장되었습니다."
                            })
                        
                        return result
                    except Exception as e:
                        logger.error(f"Error inserting document: {str(e)}")
                        return {
                            "status": "error",
                            "result_code": "F000006",
                            "message": f"문서 저장 중 오류가 발생했습니다: {str(e)}",
                            "title": doc.get('title', 'unknown')
                        }
                
                # 문서 단위 병렬 처리 실행
                max_document_threads = min(
                    int(os.getenv('INSERT_DOCUMENT_THREADS', '5')),  # 기본값: 5
                    len(docs)  # 문서 수보다 많은 스레드는 불필요
                )
                logger.info(f"[TIMING] 도메인 '{domain}'의 문서 병렬 처리 시작: {len(docs)}개 문서, 최대 {max_document_threads}개 스레드")
                
                # 문서 처리를 병렬로 수행
                doc_results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_document_threads) as executor:
                    # 각 문서에 대한 처리 작업 제출
                    future_to_doc = {executor.submit(process_document, doc): doc for doc in docs}
                    
                    # 결과 수집
                    for future in concurrent.futures.as_completed(future_to_doc):
                        try:
                            result = future.result()
                            doc_results.append(result)
                            
                            # 상태 카운터 업데이트
                            if result["status"] == "success":
                                status_counts["success"] += 1
                            elif result["status"] == "updated":
                                status_counts["updated"] += 1
                            elif result["status"] == "skipped":
                                status_counts["skipped"] += 1
                            elif result["status"] == "error":
                                status_counts["error"] += 1
                            
                            # 결과 목록에 추가
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Error getting result: {str(e)}")
                            # 오류 처리
                            error_result = {
                                "status": "error",
                                "result_code": "F000006",
                                "message": f"문서 처리 결과 수집 중 오류: {str(e)}",
                                "domain": domain
                            }
                            results.append(error_result)
                            status_counts["error"] += 1
                
                logger.info(f"[TIMING] 도메인 '{domain}'의 문서 처리 완료: {len(doc_results)}개 결과")
                
            except Exception as e:
                logger.error(f"Error processing domain {domain}: {str(e)}")
                # 도메인 처리 실패 시 해당 도메인의 모든 문서를 오류로 처리
                for doc in docs:
                    error_result = {
                        "status": "error",
                        "result_code": "F000006",
                        "message": f"도메인 처리 중 오류 발생: {str(e)}",
                        "title": doc.get('title', 'unknown'),
                        "domain": domain
                    }
                    results.append(error_result)
                    status_counts["error"] += 1
        
        # 전체 상태 결정
        if status_counts["error"] == len(request_data["documents"]):
            overall_status = "error"  # 모두 실패
            status_code = 500
        elif status_counts["error"] == 0 and status_counts["skipped"] == 0 and status_counts["updated"] == 0:
            overall_status = "success"  # 모두 새로 성공
            status_code = 200
        elif status_counts["error"] == 0:
            overall_status = "partial_success"  # 일부는 성공/업데이트/건너뜀
            status_code = 207  # Multi-Status
        else:
            overall_status = "partial_error"  # 일부 실패
            status_code = 207  # Multi-Status
        
        return jsonify({
            "status": overall_status,
            "message": f"총 {len(request_data['documents'])}개 문서 중 {status_counts['success']}개 성공, {status_counts['updated']}개 업데이트, {status_counts['skipped']}개 건너뜀, {status_counts['error']}개 실패",
            "status_counts": status_counts,
            "results": results
        }), status_code
        
    except Exception as e:
        logger.error(f"Error in insert endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "result_code": "F000007",
            "message": f"요청 처리 중 오류가 발생했습니다: {str(e)}"
        }), 500

@app.route('/rag/delete', methods=['DELETE'])
def delete_data():
    '''
    data: {
        "domain": "news",  # 도메인(컬렉션) 이름
        "doc_id": "20220804-메타버스 뉴스-삼성전자"  # 문서 ID
    }
    '''
    doc_id = request.args.get('doc_id')
    doc_domain = request.args.get('domain')
    
    # 필수 파라미터 체크
    if not doc_id:
        return jsonify({
            "error": "doc_id is required",
            "message": "문서 ID(doc_id)는 필수 입력값입니다."
        }), 400
    if not doc_domain:
        return jsonify({
            "error": "domain is required",
            "message": "도메인(domain)은 필수 입력값입니다."
        }), 400
        
    try:
        interact_manager.delete_data(doc_domain, doc_id)
        return jsonify({
            "result_code": "F000000",
            "message": "문서가 성공적으로 삭제되었습니다.",
            "doc_id": doc_id,
            "domain": doc_domain
        }), 200
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return jsonify({
            "result_code": "F000001",
            "message": f"문서 삭제 중 오류가 발생했습니다: {str(e)}",
            "doc_id": doc_id,
            "domain": doc_domain
        }), 500

@app.route('/rag/document', methods=['POST'])
def get_document():
    '''
    문서 조회 API
    
    Request Body:
    {
        "doc_id": "문서ID",
        "domains": ["도메인1", "도메인2"],  # 검색할 도메인 리스트
        "passage_id": 1  # optional, 특정 passage만 조회할 경우
    }
    
    Response Body (전체 문서 조회 시):
    {
        "doc_id": "해시된 문서ID",
        "raw_doc_id": "원본 문서ID",
        "domain_results": {
            "도메인1": {
                "doc_id": "해시된 문서ID",
                "raw_doc_id": "원본 문서ID",
                "domain": "도메인1",
                "title": "문서 제목",
                "author": "작성자",
                "info": {},  # 추가 메타데이터
                "tags": {},  # 태그 정보
                "passages": [
                    {
                        "passage_id": "패시지ID",
                        "text": "패시지 내용",
                        "position": "패시지 순서"
                    },
                    ...
                ]
            },
            "도메인2": {
                ...
            }
        }
    }
    
    Response Body (특정 패시지 조회 시):
    {
        "doc_id": "문서ID",
        "raw_doc_id": "원본 문서ID",
        "passage_id": "패시지ID",
        "text": "패시지 내용",
        "position": "패시지 순서",
        "metadata": {
            "domain": "도메인명",
            "title": "문서 제목",
            "info": {},
            "tags": {}
        }

    }
    '''
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                "error": "invalid request",
                "message": "요청 본문이 비어있습니다."
            }), 400

        doc_id = request_data.get('doc_id')
        domains = request_data.get('domains', [])
        passage_id = request_data.get('passage_id')

        if not doc_id:
            return jsonify({
                "error": "doc_id is required",
                "message": "문서 ID는 필수 입력값입니다."
            }), 400
            
        if not domains:
            return jsonify({
                "error": "domains is required",
                "message": "검색할 도메인은 필수 입력값입니다."
            }), 400
            
        if not isinstance(domains, list):
            return jsonify({
                "error": "invalid domains format",
                "message": "domains는 배열 형식이어야 합니다."
            }), 400

        # 도메인 유효성 검증
        available_collections = milvus_db.get_list_collection()
        invalid_domains = [d for d in domains if d not in available_collections]
        if invalid_domains:
            return jsonify({
                "error": "invalid domains",
                "message": f"유효하지 않은 도메인이 포함되어 있습니다: {', '.join(invalid_domains)}",
                "available_domains": available_collections
            }), 400

        try:
            if passage_id:
                # 특정 패시지 조회
                result = interact_manager.get_specific_passage(doc_id, passage_id, domains)
                if not result:
                    return jsonify({
                        "error": "Passage not found",
                        "doc_id": doc_id,
                        "passage_id": passage_id,
                        "domains": domains,
                        "message": "요청하신 패시지를 찾을 수 없습니다."
                    }), 404
                return Response(json.dumps(result, ensure_ascii=False), content_type="application/json; charset=utf-8")
            else:
                # 문서의 모든 패시지 조회
                result = interact_manager.get_document_passages(doc_id, domains)
                if not result:
                    return jsonify({
                        "error": "Document not found",
                        "doc_id": doc_id,
                        "domains": domains,
                        "message": "요청하신 문서를 찾을 수 없습니다."
                    }), 404
                return Response(json.dumps(result, ensure_ascii=False), content_type="application/json; charset=utf-8")
        except Exception as e:
            logger.error(f"Error retrieving document: {str(e)}")
            return jsonify({
                "error": "Internal server error",
                "message": "문서 조회 중 오류가 발생했습니다."
            }), 500
    except Exception as e:
        logger.error(f"Error parsing request: {str(e)}")
        return jsonify({
            "error": "Invalid request",
            "message": f"잘못된 요청입니다: {str(e)}"
        }), 400

@app.route('/rag/domains', methods=['GET'])
def get_domains():
    """
    시스템에 등록된 모든 도메인(컬렉션) 목록을 반환합니다.
    
    Returns:
        JSON: 도메인 목록과 각 도메인의 엔티티 수를 포함한 응답
    """
    try:
        # 모든 컬렉션(도메인) 목록 가져오기
        domains = milvus_db.get_list_collection()
        
        # 각 도메인별 정보 수집 (선택적)
        domain_info = []
        for domain in domains:
            try:
                # 컬렉션 정보 조회
                milvus_db.get_collection_info(domain)
                
                # 문서 개수 조회 (doc_id 기준으로 고유 개수 계산)
                collection = Collection(domain)
                
                # 도메인별 문서 개수 계산 최적화 함수
                doc_count = calculate_domain_document_count(collection, domain)
                
                # 정보 수집 및 포맷팅
                info = {
                    "name": domain,
                    "entity_count": collection.num_entities if hasattr(collection, 'num_entities') else 0,
                    "document_count": doc_count
                }
                domain_info.append(info)
                
            except Exception as e:
                logger.error(f"도메인 '{domain}' 정보 조회 실패: {str(e)}")
                # 오류 발생 시에도 기본 정보는 제공
                domain_info.append({
                    "name": domain,
                    "entity_count": 0,
                    "document_count": "계산 불가"
                })
        
        # 전체 응답 구성
        return jsonify({
            "result_code": "S000000",
            "message": "도메인 목록을 성공적으로 조회했습니다.",
            "domains": domain_info
        }), 200
        
    except Exception as e:
        logger.error(f"도메인 목록 조회 중 오류: {str(e)}")
        return jsonify({
            "result_code": "F000001",
            "message": f"도메인 목록 조회 중 오류가 발생했습니다: {str(e)}",
            "domains": []
        }), 500

def calculate_domain_document_count(collection, domain_name):
    """
    Query Iterator를 사용하여 도메인(컬렉션)의 고유 문서 개수를 계산합니다.
    대용량 컬렉션에서도 효율적으로 작동합니다.
    
    Args:
        collection: 컬렉션 객체
        domain_name: 도메인 이름
    
    Returns:
        int 또는 str: 문서 개수 또는 계산 불가 메시지
    """
    try:
        # 컬렉션 크기 확인
        collection_size = collection.num_entities if hasattr(collection, 'num_entities') else 0
        logger.info(f"도메인 '{domain_name}'의 전체 엔티티 수: {collection_size}")
        
        if collection_size == 0:
            return 0
            
        # 준비: 고유 doc_id 집합과 타이머 초기화
        unique_doc_ids = set()
        start_time = time.time()
        
        try:
            # 안전한 파라미터 설정
            batch_size = 5000  # 배치 크기
            
            # Query Iterator 사용 (페이지네이션 자동 처리)
            from pymilvus import MilvusException
            
            try:
                # 쿼리 설정
                search_params = {
                    "metric_type": "L2",
                    "params": {"nprobe": 10}
                }
                
                # 효율적인 iterator 쿼리 생성
                iterator = collection.query_iterator(
                    expr="",  # 모든 레코드 대상
                    output_fields=["doc_id"],  # doc_id만 가져옴
                    batch_size=batch_size,
                    limit=-1  # 모든 결과 반환
                )
                
                # 배치 단위로 결과 처리
                total_processed = 0
                batch_count = 0
                
                while True:
                    try:
                        # 다음 배치 가져오기
                        batch_data = iterator.next()
                        batch_count += 1
                        
                        if not batch_data:
                            break  # 더 이상 데이터가 없음
                            
                        # 배치의 고유 doc_id 추출
                        for item in batch_data:
                            if "doc_id" in item:
                                unique_doc_ids.add(item["doc_id"])
                                
                        # 처리 진행 상황 로깅
                        total_processed += len(batch_data)
                        
                        if batch_count % 10 == 0:  # 10개 배치마다 로그 출력
                            logger.info(f"도메인 '{domain_name}' 문서 수 계산 중: {len(unique_doc_ids)}개 고유 문서 / {total_processed}개 항목 처리 ({batch_count}개 배치)")
                        
                    except StopIteration:
                        # Iterator 종료
                        break
                        
                # 고유 문서 수 계산 완료
                doc_count = len(unique_doc_ids)
                end_time = time.time()
                
                # 처리 결과 로깅
                logger.info(f"도메인 '{domain_name}'의 고유 문서 수: {doc_count} (전체 엔티티: {collection_size}, 페이지 수: {batch_count}, 소요시간: {end_time - start_time:.4f}초)")
                return doc_count
                
            except MilvusException as me:
                # Milvus 오류 처리
                logger.warning(f"도메인 '{domain_name}'의 문서 개수 조회 실패: {me}")
                return "계산 불가"
                
        except Exception as e:
            # 기타 오류 처리
            logger.warning(f"도메인 '{domain_name}'의 문서 개수 계산 중 오류: {str(e)}")
            return "계산 불가"
            
    except Exception as e:
        # 전체 오류 처리
        logger.error(f"도메인 '{domain_name}'의 문서 개수 계산 중 예외 발생: {str(e)}")
        return "계산 불가"

@app.route('/rag/domains/delete', methods=['POST'])
def delete_domains():
    """
    지정된 도메인(컬렉션)을 완전히 삭제합니다.
    단일 도메인 또는 도메인 목록을 받아 해당 도메인의 모든 엔티티와 컬렉션 자체를 삭제합니다.
    
    Request Body:
    {
        "domains": ["도메인1", "도메인2"] 또는 "도메인명"
    }
    
    Returns:
        JSON: 삭제 결과 정보
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                "result_code": "F000001",
                "message": "요청 본문이 비어있습니다."
            }), 400

        domains_to_delete = request_data.get('domains')
        if domains_to_delete is None:
            return jsonify({
                "result_code": "F000002",
                "message": "domains 필드는 필수입니다."
            }), 400
            
        # 문자열로 단일 도메인이 입력된 경우 리스트로 변환
        if isinstance(domains_to_delete, str):
            domains_to_delete = [domains_to_delete]
        elif not isinstance(domains_to_delete, list):
            return jsonify({
                "result_code": "F000003",
                "message": "domains는 문자열 또는 문자열 배열이어야 합니다."
            }), 400
            
        # 빈 리스트 확인
        if len(domains_to_delete) == 0:
            return jsonify({
                "result_code": "F000004",
                "message": "삭제할 도메인이 지정되지 않았습니다."
            }), 400

        # 도메인 유효성 검증
        available_collections = milvus_db.get_list_collection()
        invalid_domains = [d for d in domains_to_delete if d not in available_collections]
        non_existent_domains = invalid_domains
        
        # 존재하지 않는 도메인 필터링
        domains_to_delete = [d for d in domains_to_delete if d in available_collections]
        
        # 삭제 결과 저장
        results = []
        
        # 각 도메인별 삭제 수행
        for domain in domains_to_delete:
            try:
                logger.info(f"도메인 '{domain}' 삭제 시작")
                
                # 컬렉션 정보 조회 (삭제 전 로깅용)
                try:
                    milvus_db.get_collection_info(domain)
                    entity_count = milvus_db.num_entities if hasattr(milvus_db, 'num_entities') else 0
                except:
                    entity_count = "알 수 없음"
                
                # 컬렉션 삭제
                milvus_db.delete_collection(domain)
                
                # 결과 추가
                results.append({
                    "name": domain,
                    "status": "success",
                    "entity_count": entity_count,
                    "message": "도메인이 성공적으로 삭제되었습니다."
                })
                logger.info(f"도메인 '{domain}' 삭제 완료 (엔티티 수: {entity_count})")
                
            except Exception as e:
                logger.error(f"도메인 '{domain}' 삭제 중 오류 발생: {str(e)}")
                results.append({
                    "name": domain,
                    "status": "error",
                    "message": f"삭제 중 오류 발생: {str(e)}"
                })
        
        # 존재하지 않는 도메인 결과에 추가
        for domain in non_existent_domains:
            results.append({
                "name": domain,
                "status": "not_found",
                "message": "존재하지 않는 도메인입니다."
            })
            
        # 전체 성공/실패 상태 결정
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        not_found_count = sum(1 for r in results if r["status"] == "not_found")
        
        if len(domains_to_delete) == 0:
            overall_status = "not_found"  # 모든 도메인이 존재하지 않음
            status_code = 404
        elif error_count == 0:
            overall_status = "success"  # 모든 삭제 성공
            status_code = 200
        elif success_count == 0:
            overall_status = "error"  # 모든 삭제 실패
            status_code = 500
        else:
            overall_status = "partial"  # 일부 성공, 일부 실패
            status_code = 207  # Multi-Status
            
        return jsonify({
            "result_code": "S000000" if overall_status == "success" else "F000005",
            "status": overall_status,
            "message": f"총 {len(results)}개 도메인 중 {success_count}개 삭제 성공, {error_count}개 실패, {not_found_count}개 없음",
            "results": results
        }), status_code
        
    except Exception as e:
        logger.error(f"도메인 삭제 중 오류 발생: {str(e)}")
        return jsonify({
            "result_code": "F000010",
            "message": f"도메인 삭제에 실패했습니다: {str(e)}",
            "status": "error"
        }), 500

@app.route("/rag/", methods=["GET"])
def index():
    print(f"hello results")
    return jsonify({"message": "Hello, FastCGI is working!"})

@app.route("/rag/test/duplicate_check", methods=["POST"])
def test_duplicate_check():
    """
    여러 doc_id에 대한 중복 체크 테스트를 수행합니다.
    
    Request Body:
    {
        "doc_ids": ["id1", "id2", "id3", ...],
        "domain": "도메인명"
    }
    
    Returns:
        JSON: 중복 체크 결과
    """
    try:
        # 로그 설정
        import logging
        test_logger = logging.getLogger('duplication')
        if not test_logger.handlers:
            log_dir = "/var/log/rag" if os.path.exists("/var/log/rag") else "../logs"
            os.makedirs(log_dir, exist_ok=True)
            handler = logging.FileHandler(os.path.join(log_dir, 'duplication.log'))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            test_logger.setLevel(logging.INFO)
            test_logger.addHandler(handler)
            test_logger.propagate = False
        
        # 요청 데이터 검증
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "요청 본문이 비어있습니다."}), 400
            
        doc_ids = request_data.get("doc_ids")
        domain = request_data.get("domain")
        
        if not doc_ids:
            return jsonify({"error": "doc_ids 필드는 필수입니다."}), 400
            
        if not domain:
            return jsonify({"error": "domain 필드는 필수입니다."}), 400
            
        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]  # 단일 ID를 리스트로 변환
            
        # 도메인 유효성 검증
        available_collections = milvus_db.get_list_collection()
        if domain not in available_collections:
            return jsonify({
                "error": f"도메인 '{domain}'이 존재하지 않습니다.",
                "available_domains": available_collections
            }), 404
            
        # 테스트 시작 로깅
        test_logger.info(f"중복 체크 테스트 시작: {len(doc_ids)}개 문서, 도메인: {domain}")
        test_logger.info(f"테스트할 문서 ID: {doc_ids}")
        
        # 중복 체크 수행
        start_time = time.time()
        try:
            results = interact_manager.check_duplicates(doc_ids, domain)
            end_time = time.time()
            
            # 결과 로깅
            test_logger.info(f"중복 체크 결과: {results}")
            test_logger.info(f"소요 시간: {end_time - start_time:.4f}초")
            
            # 응답 구성
            return jsonify({
                "status": "success",
                "message": f"{len(doc_ids)}개 문서 중 {len(results)}개 중복 발견",
                "test_doc_ids": doc_ids,
                "duplicate_doc_ids": results,
                "domain": domain,
                "duration": end_time - start_time
            }), 200
            
        except Exception as e:
            test_logger.error(f"중복 체크 오류: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"중복 체크 중 오류 발생: {str(e)}",
                "test_doc_ids": doc_ids,
                "domain": domain
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"요청 처리 중 오류 발생: {str(e)}"
        }), 500

@app.route("/rag/test/in_operator", methods=["POST"])
def test_in_operator():
    """
    IN 연산자를 직접 테스트하는 엔드포인트입니다.
    
    Request Body:
    {
        "doc_ids": ["id1", "id2", "id3", ...],
        "domain": "도메인명",
        "batch_size": 20  // 선택적, 기본값은 20
    }
    
    Returns:
        JSON: IN 연산자 테스트 결과
    """
    try:
        # 로그 설정
        import logging
        test_logger = logging.getLogger('duplication')
        if not test_logger.handlers:
            log_dir = "/var/log/rag" if os.path.exists("/var/log/rag") else "../logs"
            os.makedirs(log_dir, exist_ok=True)
            handler = logging.FileHandler(os.path.join(log_dir, 'duplication.log'))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            test_logger.setLevel(logging.INFO)
            test_logger.addHandler(handler)
            test_logger.propagate = False
        
        # 요청 데이터 검증
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "요청 본문이 비어있습니다."}), 400
            
        doc_ids = request_data.get("doc_ids")
        domain = request_data.get("domain")
        batch_size = request_data.get("batch_size", 20)  # 기본값 20
        
        if not doc_ids:
            return jsonify({"error": "doc_ids 필드는 필수입니다."}), 400
            
        if not domain:
            return jsonify({"error": "domain 필드는 필수입니다."}), 400
            
        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]  # 단일 ID를 리스트로 변환
            
        # 도메인 유효성 검증
        available_collections = milvus_db.get_list_collection()
        if domain not in available_collections:
            return jsonify({
                "error": f"도메인 '{domain}'이 존재하지 않습니다.",
                "available_domains": available_collections
            }), 404
            
        # 테스트 시작 로깅
        test_logger.info(f"IN 연산자 테스트 시작: {len(doc_ids)}개 문서, 도메인: {domain}, 배치 크기: {batch_size}")
        
        # 컬렉션 로드
        collection = Collection(domain)
        collection.load()
        test_logger.info(f"컬렉션 '{domain}' 로드 완료")
        
        # 결과 저장 변수
        all_results = []
        batch_results = []
        found_doc_ids = []
        
        # 배치 단위로 처리
        start_time = time.time()
        total_query_time = 0
        
        try:
            for i in range(0, len(doc_ids), batch_size):
                batch = doc_ids[i:i + batch_size]
                if not batch:
                    continue
                    
                # IN 연산자를 사용하여 배치로 쿼리
                ids_str = ", ".join([f'"{id}"' for id in batch])
                expr = f'doc_id in [{ids_str}]'  # 올바른 IN 연산자 형식 사용
                
                batch_start = time.time()
                test_logger.info(f"배치 {i//batch_size + 1} 쿼리 실행: {expr[:100]}{'...' if len(expr) > 100 else ''}")
                
                try:
                    # 중요: 여기서 expr 매개변수로 쿼리 실행
                    results = collection.query(
                        expr=expr,
                        output_fields=["doc_id", "passage_id"],
                        limit=10000  # 충분히 큰 값으로 설정
                    )
                    
                    batch_end = time.time()
                    batch_time = batch_end - batch_start
                    total_query_time += batch_time
                    
                    # 결과 로깅
                    result_doc_ids = {result["doc_id"] for result in results if "doc_id" in result}
                    found_doc_ids.extend(list(result_doc_ids))
                    
                    batch_results.append({
                        "batch_index": i//batch_size + 1,
                        "batch_size": len(batch),
                        "query_time": batch_time,
                        "result_count": len(results),
                        "unique_doc_ids": len(result_doc_ids),
                        "found_ids": list(result_doc_ids)
                    })
                    
                    test_logger.info(f"배치 {i//batch_size + 1} 쿼리 결과: {len(results)}개 항목, {len(result_doc_ids)}개 고유 문서, 소요시간: {batch_time:.4f}초")
                    
                except Exception as e:
                    test_logger.error(f"배치 {i//batch_size + 1} 처리 중 오류: {str(e)}")
                    batch_results.append({
                        "batch_index": i//batch_size + 1,
                        "batch_size": len(batch),
                        "error": str(e),
                        "status": "error"
                    })
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 중복 제거된 최종 결과
            unique_found_doc_ids = list(set(found_doc_ids))
            
            test_logger.info(f"IN 연산자 테스트 완료: 총 {len(doc_ids)}개 문서 중 {len(unique_found_doc_ids)}개 발견, 총 소요시간: {total_time:.4f}초, 쿼리 시간: {total_query_time:.4f}초")
            
            # 응답 구성
            return jsonify({
                "status": "success",
                "message": f"{len(doc_ids)}개 문서 중 {len(unique_found_doc_ids)}개 발견",
                "test_doc_ids": doc_ids,
                "found_doc_ids": unique_found_doc_ids,
                "domain": domain,
                "batch_size": batch_size,
                "total_time": total_time,
                "query_time": total_query_time,
                "batch_results": batch_results
            }), 200
            
        except Exception as e:
            test_logger.error(f"IN 연산자 테스트 오류: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"IN 연산자 테스트 중 오류 발생: {str(e)}",
                "test_doc_ids": doc_ids,
                "domain": domain
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"요청 처리 중 오류 발생: {str(e)}"
        }), 500

@app.route("/rag/test/or_operator", methods=["POST"])
def test_or_operator():
    """
    OR 연산자를 직접 테스트하는 엔드포인트입니다.
    
    Request Body:
    {
        "doc_ids": ["id1", "id2", "id3", ...],
        "domain": "도메인명",
        "batch_size": 20,  // 선택적, 기본값은 20
        "use_sample_ids": true  // 선택적, 설정 시 컬렉션에서 샘플 ID를 가져와 테스트
    }
    
    Returns:
        JSON: OR 연산자 테스트 결과
    """
    try:
        # 로그 설정
        import logging
        test_logger = logging.getLogger('duplication')
        if not test_logger.handlers:
            log_dir = "/var/log/rag" if os.path.exists("/var/log/rag") else "../logs"
            os.makedirs(log_dir, exist_ok=True)
            handler = logging.FileHandler(os.path.join(log_dir, 'duplication.log'))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            test_logger.setLevel(logging.INFO)
            test_logger.addHandler(handler)
            test_logger.propagate = False
        
        # 요청 데이터 검증
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "요청 본문이 비어있습니다."}), 400
            
        doc_ids = request_data.get("doc_ids")
        domain = request_data.get("domain")
        batch_size = request_data.get("batch_size", 20)  # 기본값 20
        use_sample_ids = request_data.get("use_sample_ids", False)  # 샘플 ID 사용 여부
        
        if not domain:
            return jsonify({"error": "domain 필드는 필수입니다."}), 400
            
        # 도메인 유효성 검증
        available_collections = milvus_db.get_list_collection()
        if domain not in available_collections:
            return jsonify({
                "error": f"도메인 '{domain}'이 존재하지 않습니다.",
                "available_domains": available_collections
            }), 404
        
        # 컬렉션 로드
        collection = Collection(domain)
        collection.load()
        test_logger.info(f"컬렉션 '{domain}' 로드 완료")
        
        # 컬렉션 스키마 로깅 (디버깅용)
        schema = collection.schema
        test_logger.info(f"컬렉션 스키마: {schema}")
        test_logger.info(f"필드 이름 목록: {[field.name for field in schema.fields]}")
        
        # 실제 문서 ID 조회 (요청한 코드 녹임)
        try:
            # 데이터베이스에서 실제 문서 ID 5개 샘플링
            sample_docs = collection.query(
                expr="",  # 모든 문서
                output_fields=["doc_id"],
                limit=5
            )
            test_logger.info(f"실제 문서 ID 샘플 (5개): {sample_docs}")
            
            if not sample_docs:
                test_logger.warning(f"컬렉션 '{domain}'에 문서가 없습니다!")
                return jsonify({
                    "status": "error",
                    "message": f"컬렉션 '{domain}'에 문서가 없습니다.",
                    "domain": domain
                }), 400
            
            # 첫 번째 문서로 단일 조회 테스트 (디버깅용)
            if sample_docs and len(sample_docs) > 0:
                first_doc_id = sample_docs[0]["doc_id"]
                test_logger.info(f"첫 번째 문서 ID: {first_doc_id}")
                
                single_test = collection.query(
                    expr=f'doc_id == "{first_doc_id}"',
                    output_fields=["doc_id", "passage_id"],
                    limit=1
                )
                test_logger.info(f"단일 ID 조회 테스트 결과: {single_test}")
            
            # 샘플 ID 사용 옵션이 켜져 있으면 실제 컬렉션의 ID 사용
            if use_sample_ids and sample_docs:
                doc_ids = [doc["doc_id"] for doc in sample_docs if "doc_id" in doc]
                test_logger.info(f"샘플 ID로 대체: {doc_ids}")
        except Exception as e:
            test_logger.error(f"샘플 문서 조회 실패: {str(e)}")
            
        # 요청된 doc_ids 확인
        if not doc_ids and not use_sample_ids:
            return jsonify({"error": "doc_ids 필드는 필수입니다. 또는 use_sample_ids=true를 설정하세요."}), 400
            
        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]  # 단일 ID를 리스트로 변환
            
        # 테스트 시작 로깅
        test_logger.info(f"OR 연산자 테스트 시작: {len(doc_ids)}개 문서, 도메인: {domain}, 배치 크기: {batch_size}")
        test_logger.info(f"테스트할 doc_ids: {doc_ids}")
        
        # 결과 저장 변수
        all_results = []
        batch_results = []
        found_doc_ids = []
        
        # 배치 단위로 처리
        start_time = time.time()
        total_query_time = 0
        
        try:
            for i in range(0, len(doc_ids), batch_size):
                batch = doc_ids[i:i + batch_size]
                if not batch:
                    continue
                    
                # OR 연산자를 사용하여 배치로 쿼리
                conditions = []
                for doc_id in batch:
                    conditions.append(f'doc_id == "{doc_id}"')
                
                expr = " || ".join(conditions)  # OR 연산자 사용
                
                batch_start = time.time()
                test_logger.info(f"배치 {i//batch_size + 1} 쿼리 실행: {expr[:100]}{'...' if len(expr) > 100 else ''}")
                
                try:
                    # 중요: 여기서 expr 매개변수로 쿼리 실행
                    results = collection.query(
                        expr=expr,
                        output_fields=["doc_id", "passage_id"],
                        limit=10000  # 충분히 큰 값으로 설정
                    )
                    
                    batch_end = time.time()
                    batch_time = batch_end - batch_start
                    total_query_time += batch_time
                    
                    # 결과 로깅
                    result_doc_ids = {result["doc_id"] for result in results if "doc_id" in result}
                    found_doc_ids.extend(list(result_doc_ids))
                    
                    # 디버깅: 결과 객체 타입 확인
                    test_logger.info(f"쿼리 결과 유형: {type(results).__name__}")
                    test_logger.info(f"결과 첫 항목 샘플 (있는 경우): {results[0] if results else 'N/A'}")
                    
                    batch_results.append({
                        "batch_index": i//batch_size + 1,
                        "batch_size": len(batch),
                        "query_time": batch_time,
                        "result_count": len(results),
                        "unique_doc_ids": len(result_doc_ids),
                        "found_ids": list(result_doc_ids)
                    })
                    
                    test_logger.info(f"배치 {i//batch_size + 1} 쿼리 결과: {len(results)}개 항목, {len(result_doc_ids)}개 고유 문서, 소요시간: {batch_time:.4f}초")
                    
                except Exception as e:
                    test_logger.error(f"배치 {i//batch_size + 1} 처리 중 오류: {str(e)}")
                    test_logger.error(f"오류 발생 쿼리: {expr}")
                    
                    # 대체 방법 시도: 개별 쿼리
                    test_logger.info("대체 방법으로 개별 쿼리 시도...")
                    
                    individual_results = []
                    for doc_id in batch[:3]:  # 처음 3개만 시도
                        try:
                            single_expr = f'doc_id == "{doc_id}"'
                            single_result = collection.query(
                                expr=single_expr,
                                output_fields=["doc_id", "passage_id"],
                                limit=1
                            )
                            test_logger.info(f"개별 쿼리 '{single_expr}' 결과: {single_result}")
                            individual_results.append({
                                "doc_id": doc_id,
                                "result": single_result
                            })
                        except Exception as single_err:
                            test_logger.error(f"개별 쿼리 오류: {str(single_err)}")
                    
                    batch_results.append({
                        "batch_index": i//batch_size + 1,
                        "batch_size": len(batch),
                        "error": str(e),
                        "status": "error",
                        "individual_tests": individual_results
                    })
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 중복 제거된 최종 결과
            unique_found_doc_ids = list(set(found_doc_ids))
            
            test_logger.info(f"OR 연산자 테스트 완료: 총 {len(doc_ids)}개 문서 중 {len(unique_found_doc_ids)}개 발견, 총 소요시간: {total_time:.4f}초, 쿼리 시간: {total_query_time:.4f}초")
            
            # 응답 구성
            return jsonify({
                "status": "success",
                "message": f"{len(doc_ids)}개 문서 중 {len(unique_found_doc_ids)}개 발견",
                "test_doc_ids": doc_ids,
                "found_doc_ids": unique_found_doc_ids,
                "sample_docs": sample_docs,  # 실제 DB에서 조회한 샘플 문서 추가
                "domain": domain,
                "batch_size": batch_size,
                "total_time": total_time,
                "query_time": total_query_time,
                "batch_results": batch_results,
                "schema_fields": [field.name for field in schema.fields] if 'schema' in locals() else []
            }), 200
            
        except Exception as e:
            test_logger.error(f"OR 연산자 테스트 오류: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"OR 연산자 테스트 중 오류 발생: {str(e)}",
                "test_doc_ids": doc_ids,
                "domain": domain,
                "sample_docs": sample_docs if 'sample_docs' in locals() else []  # 조회에 성공했다면 샘플 문서 포함
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"요청 처리 중 오류 발생: {str(e)}"
        }), 500

if __name__ == "__main__":
    print(f"Start results")
    app.run(host="0.0.0.0", port=5000)