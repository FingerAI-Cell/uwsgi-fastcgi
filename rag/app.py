from flask import Flask, send_file, request, jsonify, Response
from pymilvus import Collection
from dotenv import load_dotenv
from src import EnvManager, InteractManager
import logging
import json 
import os 

logger = logging.getLogger('api_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('api-result.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

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

@app.route('/data/show', methods=['GET'])
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

@app.route('/search', methods=['GET'])
def search_data():
    # 기본 검색 파라미터
    query_text = request.args.get('query_text')
    top_k = request.args.get('top_k', 5)
    domain = request.args.get('domain')  # 선택적 도메인 필터

    # 추가 필터링 파라미터
    author = request.args.get('author')  # 작성자 필터 추가
    start_date = request.args.get('start_date')  # YYYYMMDD 형식
    end_date = request.args.get('end_date')      # YYYYMMDD 형식
    title_query = request.args.get('title')      # 제목 검색
    info_filter = request.args.get('info_filter') # JSON 형식의 info 필터 조건
    tags_filter = request.args.get('tags_filter') # JSON 형식의 tags 필터 조건

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
    
    # 필터 조건 파싱
    filter_conditions = {}
    if domain:
        filter_conditions['domain'] = domain
    if author:  # 작성자 필터 추가
        filter_conditions['author'] = author
    if start_date or end_date:
        filter_conditions['date_range'] = {'start': start_date, 'end': end_date}
    if title_query:
        filter_conditions['title'] = title_query
    if info_filter:
        try:
            filter_conditions['info'] = json.loads(info_filter)
        except json.JSONDecodeError:
            return jsonify({
                "result_code": "F000003",
                "message": "info_filter는 유효한 JSON 형식이어야 합니다.",
                "search_result": None
            }), 400
    if tags_filter:
        try:
            filter_conditions['tags'] = json.loads(tags_filter)
        except json.JSONDecodeError:
            return jsonify({
                "result_code": "F000004",
                "message": "tags_filter는 유효한 JSON 형식이어야 합니다.",
                "search_result": None
            }), 400
    
    try:
        search_results = interact_manager.retrieve_data(
            query_text, 
            top_k, 
            filter_conditions=filter_conditions
        )
        
        response_data = {
            "result_code": "F000000",
            "message": "검색이 성공적으로 완료되었습니다.",
            "search_params": {
                "query_text": query_text,
                "top_k": top_k,
                "filters": filter_conditions
            },
            "search_result": search_results
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

@app.route('/insert', methods=['POST'])
def insert_data():
    '''
    doc_id: yyyymmdd-title-author   e.g) 20240301-메타버스 뉴스-삼성전자
    data: {
        "domain": "news"   - collection_name 
        "title": "메타버스 뉴스"
        "author": "삼성전자"  # 작성자 (기업 또는 특정인물)
        "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다 ... "
        "info": {
            "press_num": "비즈니스 워치"
            "url": "http://~"
        }
        "tags": {
            "date": "20220804"
            "user": "user01"
        }
    }
    '''
    data = request.json
    doc_id = f"{data['tags']['date'].replace('-','')}-{data['title']}-{data['author']}"
    if data['domain'] not in milvus_db.get_list_collection():
        interact_manager.create_domain(data['domain'])
    interact_manager.insert_data(data['domain'], doc_id, data['title'], data['author'], data['text'], data['info'], data['tags'])
    return jsonify({"status": "received"}), 200

@app.route('/delete', methods=['DELETE'])
def delete_data():
    '''
    data: {
        "date": "20220804"
        "title": "메타버스%20뉴스"
        "author": "삼성전자"  # 작성자 추가
        "domain": "news"
    }
    '''
    doc_date = request.args.get('date')
    doc_title = request.args.get('title')
    doc_author = request.args.get('author')
    doc_domain = request.args.get('domain')
    
    # 필수 파라미터 체크
    if not doc_date:
        return jsonify({
            "error": "date is required",
            "message": "날짜(date)는 필수 입력값입니다."
        }), 400
    if not doc_title:
        return jsonify({
            "error": "title is required",
            "message": "제목(title)은 필수 입력값입니다."
        }), 400
    if not doc_author:
        return jsonify({
            "error": "author is required",
            "message": "작성자(author)는 필수 입력값입니다."
        }), 400
    if not doc_domain:
        return jsonify({
            "error": "domain is required",
            "message": "도메인(domain)은 필수 입력값입니다."
        }), 400
        
    doc_id = f"{doc_date.replace('-','')}-{doc_title}-{doc_author}"
    interact_manager.delete_data(doc_domain, doc_id)
    return jsonify({"status": "received"}), 200

@app.route('/document', methods=['GET'])
def get_document():
    doc_id = request.args.get('doc_id')
    passage_id = request.args.get('passage_id')

    if not doc_id:
        return jsonify({
            "error": "doc_id is required",
            "message": "문서 ID는 필수 입력값입니다."
        }), 400

    try:
        if passage_id:
            # 특정 패시지 조회
            result = interact_manager.get_specific_passage(doc_id, passage_id)
            if not result:
                return jsonify({
                    "error": "Passage not found",
                    "doc_id": doc_id,
                    "passage_id": passage_id,
                    "message": "요청하신 패시지를 찾을 수 없습니다."
                }), 404
            return Response(json.dumps(result, ensure_ascii=False), content_type="application/json; charset=utf-8")
        else:
            # 문서의 모든 패시지 조회
            result = interact_manager.get_document_passages(doc_id)
            if not result:
                return jsonify({
                    "error": "Document not found",
                    "doc_id": doc_id,
                    "message": "요청하신 문서를 찾을 수 없습니다."
                }), 404
            return Response(json.dumps(result, ensure_ascii=False), content_type="application/json; charset=utf-8")
    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "문서 조회 중 오류가 발생했습니다."
        }), 500

@app.route("/", methods=["GET"])
def index():
    print(f"hello results")
    return jsonify({"message": "Hello, FastCGI is working!"})

if __name__ == "__main__":
    print(f"Start results")
    app.run(host="0.0.0.0", port=5000)