from flask import Flask, send_file, request, jsonify, Response
from pymilvus import Collection
from dotenv import load_dotenv
from src import EnvManager, InteractManager
import logging
import json 
import os 

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/var/log/rag/app.log") if os.path.exists("/var/log/rag") else logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("rag-backend")

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
args['ip_addr'] = os.getenv('ip_addr')   

env_manager = EnvManager(args)
env_manager.set_processors()
emb_model = env_manager.set_emb_model()
milvus_data, milvus_meta = env_manager.set_vectordb()
milvus_db = env_manager.milvus_db
interact_manager = InteractManager(data_p=env_manager.data_p, vectorenv=milvus_db, vectordb=milvus_data, emb_model=emb_model)

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
    
    try:
        # 각 도메인별 검색 결과 수집
        all_results = []
        searched_domains = domains if domains else [available_collections[0]]  # 도메인이 지정되지 않으면 첫 번째 도메인만 사용
        
        for domain in searched_domains:
            domain_filter_conditions = dict(filter_conditions)
            domain_filter_conditions['domain'] = domain
            
            results = interact_manager.retrieve_data(
                query_text, 
                top_k, 
                filter_conditions=domain_filter_conditions
            )
            
            # 도메인 정보 추가
            for result in results:
                result['domain'] = domain
            
            all_results.extend(results)
        
        # 전체 결과를 score 기준으로 재정렬하고 top_k개만 선택
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = all_results[:top_k]
        
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
            "search_result": cleaned_results
        }
        
        return Response(json.dumps(response_data, ensure_ascii=False), 
                      content_type="application/json; charset=utf-8")
                      
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({
            "result_code": "F000005",
            "message": f"검색 중 오류가 발생했습니다: {str(e)}",
            "search_result": None
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

        # 필수 필드 검증
        required_fields = ['domain', 'title', 'author', 'text', 'tags']
        results = []
        
        # 상태별 카운터 초기화
        status_counts = {
            "success": 0,  # 새로 삽입됨
            "skipped": 0,  # 중복으로 무시됨
            "error": 0     # 오류 발생
        }
        
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

                # doc_id 생성 및 해시 처리
                raw_doc_id = f"{doc['tags']['date'].replace('-','')}-{doc['title']}-{doc['author']}"
                doc_id = env_manager.data_p.hash_text(raw_doc_id, hash_type='blake')
                
                # 도메인이 없으면 생성
                if doc['domain'] not in milvus_db.get_list_collection():
                    interact_manager.create_domain(doc['domain'])
                    # 새로 생성된 컬렉션 로드
                    collection = Collection(doc['domain'])
                    collection.load()
                    print(f"[DEBUG] New collection {doc['domain']} created and loaded")
                
                # 데이터 삽입 시도
                insert_status = interact_manager.insert_data(
                    doc['domain'], 
                    doc_id,  # 해시된 doc_id 사용
                    doc['title'], 
                    doc['author'], 
                    doc['text'], 
                    doc.get('info', {}),
                    doc['tags'],
                    ignore=ignore  # 전체 요청에 대한 ignore 값 사용
                )
                
                if insert_status == "skipped":
                    results.append({
                        "status": "skipped",
                        "result_code": "F000000",
                        "message": "이미 존재하는 문서로 건너뛰었습니다.",
                        "doc_id": doc_id,
                        "raw_doc_id": raw_doc_id,
                        "domain": doc['domain'],
                        "title": doc['title']
                    })
                    status_counts["skipped"] += 1
                else:  # success
                    results.append({
                        "status": "success",
                        "result_code": "F000000",
                        "message": "문서가 성공적으로 저장되었습니다.",
                        "doc_id": doc_id,
                        "raw_doc_id": raw_doc_id,
                        "domain": doc['domain'],
                        "title": doc['title']
                    })
                    status_counts["success"] += 1
                
            except Exception as e:
                logger.error(f"Error inserting document: {str(e)}")
                results.append({
                    "status": "error",
                    "result_code": "F000006",
                    "message": f"문서 저장 중 오류가 발생했습니다: {str(e)}",
                    "title": doc.get('title', 'unknown')
                })
                status_counts["error"] += 1

        # 전체 상태 결정
        if status_counts["error"] == len(request_data["documents"]):
            overall_status = "error"  # 모두 실패
        elif status_counts["error"] == 0 and status_counts["skipped"] == 0:
            overall_status = "success"  # 모두 성공
        elif status_counts["error"] == 0:
            overall_status = "partial_success"  # 일부는 성공하고 일부는 건너뜀
        else:
            overall_status = "partial_error"  # 일부 실패
        
        return jsonify({
            "status": overall_status,
            "message": f"총 {len(request_data['documents'])}개 문서 중 {status_counts['success']}개 성공, {status_counts['skipped']}개 건너뜀, {status_counts['error']}개 실패",
            "status_counts": status_counts,
            "results": results
        }), 200 if overall_status != "error" else 500
        
    except Exception as e:
        logger.error(f"Error in insert endpoint: {str(e)}")
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
                
                # 도메인이 없으면 생성
                if doc['domain'] not in milvus_db.get_list_collection():
                    interact_manager.create_domain(doc['domain'])
                    # 새로 생성된 컬렉션 로드
                    collection = Collection(doc['domain'])
                    collection.load()
                    print(f"[DEBUG] New collection {doc['domain']} created and loaded")
                
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
                
                if insert_status == "skipped":
                    results.append({
                        "status": "skipped",
                        "result_code": "F000000",
                        "message": "이미 존재하는 문서로 건너뛰었습니다.",
                        "doc_id": doc['doc_id'],
                        "passage_id": doc['passage_id'],
                        "domain": doc['domain'],
                        "title": doc['title']
                    })
                    status_counts["skipped"] += 1
                elif insert_status == "updated":
                    results.append({
                        "status": "updated",
                        "result_code": "F000000",
                        "message": "기존 문서를 삭제하고 새로운 문서로 업데이트했습니다.",
                        "doc_id": doc['doc_id'],
                        "passage_id": doc['passage_id'],
                        "domain": doc['domain'],
                        "title": doc['title']
                    })
                    status_counts["updated"] += 1
                else:  # success
                    results.append({
                        "status": "success",
                        "result_code": "F000000",
                        "message": "문서가 성공적으로 저장되었습니다.",
                        "doc_id": doc['doc_id'],
                        "passage_id": doc['passage_id'],
                        "domain": doc['domain'],
                        "title": doc['title']
                    })
                    status_counts["success"] += 1
                
            except Exception as e:
                logger.error(f"Error inserting document: {str(e)}")
                results.append({
                    "status": "error",
                    "result_code": "F000006",
                    "message": f"문서 저장 중 오류가 발생했습니다: {str(e)}",
                    "title": doc.get('title', 'unknown')
                })
                status_counts["error"] += 1

        # 전체 상태 결정
        if status_counts["error"] == len(request_data["documents"]):
            overall_status = "error"  # 모두 실패
        elif status_counts["error"] == 0 and status_counts["skipped"] == 0 and status_counts["updated"] == 0:
            overall_status = "success"  # 모두 새로 성공
        elif status_counts["error"] == 0:
            overall_status = "partial_success"  # 일부는 성공/업데이트/건너뜀
        else:
            overall_status = "partial_error"  # 일부 실패
        
        return jsonify({
            "status": overall_status,
            "message": f"총 {len(request_data['documents'])}개 문서 중 {status_counts['success']}개 성공, {status_counts['updated']}개 업데이트, {status_counts['skipped']}개 건너뜀, {status_counts['error']}개 실패",
            "status_counts": status_counts,
            "results": results
        }), 200 if overall_status != "error" else 500
        
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
                
                # 정보 수집 및 포맷팅
                info = {
                    "name": domain,
                    "entity_count": milvus_db.num_entities if hasattr(milvus_db, 'num_entities') else 0
                }
                domain_info.append(info)
            except Exception as e:
                # 컬렉션 접근 오류 발생 시 기본 정보만 추가
                domain_info.append({
                    "name": domain,
                    "entity_count": 0,
                    "error": str(e)
                })
        
        return jsonify({
            "result_code": "S000000",
            "message": "도메인 목록을 성공적으로 조회했습니다.",
            "domains": domain_info
        }), 200
        
    except Exception as e:
        logger.error(f"도메인 목록 조회 중 오류 발생: {str(e)}")
        return jsonify({
            "result_code": "F000010",
            "message": f"도메인 목록 조회에 실패했습니다: {str(e)}",
            "domains": []
        }), 500

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

if __name__ == "__main__":
    print(f"Start results")
    app.run(host="0.0.0.0", port=5000)