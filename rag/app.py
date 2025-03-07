from pymilvus import Collection, CollectionSchema, FieldSchema, utility
from flask import Flask, send_file, request, jsonify, Response
from src import MilVus, DataMilVus, MilvusMeta
from src import EnvManager
import logging


logger = logging.getLogger('api_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('api-result.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = Flask(__name__)
args = dict()
args['config_path'] = "./config"
args['llm_config'] = "llm_config.json"
args['db_config'] = "db_config.json"
args['collection_name'] = "rule_book"
env_manager = EnvManager(args)
env_manager.set_config()
# print(env_manager)

emb_model = env_manager.set_emb_model()
data_milvus = env_manager.set_vectordb()
milvus_db = env_manager.milvus_db
# print(emb_model)

@app.route('/data/show', methods=['GET'])
def show_data():
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({"error": "collection_name is required"}), 400
    
    collection = Collection(collection_name)
    milvus_db.get_partition_info(collection)
    milvus_db.get_collection_info(collection_name)
    # 조회 결과를 JSON 형태로 반환
    return jsonify({
        "collection_name": collection_name,
        "partition_info": milvus_db.partition_names,
        "collection_info": milvus_db.partition_entities_num
    }), 200

@app.route('/search', methods=['GET'])
def search_data():
    '''
    data: {
        "domain": "news"
        "doc_id": "20220221412"
        "passage_id": 1
        "title": "메타버스 뉴스"
        "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다 ... "
        "info": {
            "press_num": "비즈니스 워치"
            "url": "http://~"
        }
        "tags": {
            "date": "2022-08-04"
            "user": "user01"
        }
    }
    '''
    query_text = request.args.get('query_text')
    top_k = request.args.get('top_k')
    try:
        domain = request.args.get('domain')
    except:
        pass 
    

@app.route('/insert', methods=['POST'])
def insert_data():
    '''
    data: {
        "domain": "news"
        "doc_id": "20220221412"
        "passage_id": 1
        "title": "메타버스 뉴스"
        "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다 ... "
        "info": {
            "press_num": "비즈니스 워치"
            "url": "http://~"
        }
        "tags": {
            "date": "2022-08-04"
            "user": "user01"
        }
    }
    '''
    data = request.json
    query = data['']
    return jsonify({"status": "received"}), 200

@app.route('/delete', methods=['DELETE'])
def delete_data():
    '''
    data: {
        "doc_id":
        "passage_id":
    }
    '''
    data = request.args.get('query_text')
    return jsonify({"status": "receviced"}), 200


@app.route('/add_domain', methods=['POST'])
def add_domain():
    '''
    data: {
        "domain": "finance",
        "info_schema": {
            "press_nm": "String",
            "url": "String",
            "author": "String"    
        },
        "tags_schema": {
            "date": "String",
            "category": "String"
        }
    }
    '''
    data = request.json
    domain = data['domain']
    info_schema = data['info_schema']
    tags_schema = data['tags_schema']
    return jsonify({"message": "Domain added"})


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Hello, FastCGI is working!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
