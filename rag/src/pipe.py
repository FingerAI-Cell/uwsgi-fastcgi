from dotenv import load_dotenv
from pymilvus import Collection
from .milvus import MilvusEnvManager, DataMilVus, MilvusMeta
from .data_p import DataProcessor, VectorProcessor
import json
import os

class EnvManager():
    def __init__(self, args):
        self.args = args
        load_dotenv()
        self.ip_addr = os.getenv('ip_addr')
        self.cohere_api = os.getenv('COHERE_API_KEY')   

    def set_config(self):
        with open(os.path.join(self.args['config_path'], self.args['db_config'])) as f:
            self.db_config = json.load(f)
        with open(os.path.join(self.args['config_path'], self.args['llm_config'])) as f:
            self.llm_config = json.load(f)
            
    def set_vectordb(self):
        self.db_config['ip_addr'] = self.ip_addr
        self.milvus_db = MilvusEnvManager(self.db_config)
        self.milvus_db.set_env()
        data_milvus = DataMilVus(self.db_config)
        meta_milvus = MilvusMeta()
        meta_milvus.set_rulebook_map()
        rulebook_eng_to_kor = meta_milvus.rulebook_eng_to_kor

        self.collection = Collection(self.args['collection_name'])
        self.collection.load()

        self.milvus_db.get_partition_info(self.collection)
        self.partition_list = [rulebook_eng_to_kor[p_name] for p_name in self.milvus_db.partition_names if not p_name.startswith('_')]
        return data_milvus

    def set_emb_model(self):
        emb_model = VectorProcessor()
        emb_model.set_emb_model(model_type='bge')
        emb_model.set_embbeding_config()
        return emb_model         

class InputManager:
    '''
    입력 데이터 길이가 500자 이상인 경우 500자 단위로 자름
    텍스트 벡터로 변환 
    '''
    pass