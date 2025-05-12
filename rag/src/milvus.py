from pymilvus import MilvusClient, DataType
from pymilvus import connections, db
from pymilvus import Collection, CollectionSchema, FieldSchema, utility
import logging
import ast
import re
import json

class MilVus:
    _connected = False 

    def __init__(self, db_config):
        self.db_config = db_config 
        self.ip_addr = db_config['ip_addr'] 
        self.port = db_config['port']
        self.set_env()

        if not MilVus._connected:
            self.set_env()
            MilVus._connected = True  # 연결 상태 업데이트

    def set_env(self):
        # 기존 연결이 있는지 확인
        try:
            conn = connections.get_connection("default")
            if conn is not None and conn.connected():
                print("Milvus already connected. Reusing connection.")
                return
        except Exception:
            pass  # 연결이 없으면 새로운 연결 생성
        
        # 한 가지 방식으로만 연결 (MilvusClient 사용)
        self.client = MilvusClient(
            uri="http://" + self.ip_addr + ":19530", 
            port=self.port
        )
        
        # 원래 설정대로 host 이름 유지
        self.conn = connections.connect(
            alias="default", 
            host='milvus-standalone',   # 원래 설정대로 호스트명 사용
            port=self.port
        )

    def _get_data_type(self, dtype):
        if dtype == "FLOAT_VECTOR":
            return DataType.FLOAT_VECTOR
        elif dtype == "INT64":
            return DataType.INT64
        elif dtype == "VARCHAR":
            return DataType.VARCHAR
        elif dtype == "JSON":
            return DataType.JSON  # JSON 타입 추가
        else:
            raise ValueError(f"Unsupported data type: {dtype}")

    def get_list_collection(self):
        return utility.list_collections()

    def get_partition_info(self, collection_name):
        collection = Collection(collection_name)
        self.partitions = collection.partitions 
        self.partition_names = [] 
        self.partition_entities_num = [] 
        for partition in self.partitions: 
            # print(f'partition name: {partition.name} num of entitiy: {partition.num_entities}')
            self.partition_names.append(partition.name)
            self.partition_entities_num.append(partition.num_entities)

    def get_collection_info(self, collection_name):
        collection = Collection(collection_name)
        self.collection_schema = collection.schema 
        self.collection_name = collection.name 
        self.collection_is_empty = collection.is_empty 
        self.collection_primary_key = collection.primary_field
        self.collection_partitions = collection.partition
        self.num_entities = collection.num_entities


class MilvusEnvManager(MilVus):
    def __init__(self, args):
        super().__init__(args)
        self.logger = logging.getLogger(__name__)

    def create_collection(self, collection_name, schema, shards_num):
        collection = Collection(
            name=collection_name,
            schema=schema,
            using='default',
            shards_num=shards_num
        )
        return collection 

    def create_field_schema(self, schema_name, dtype=None, dim=1024, max_length=200, is_primary=False):
        data_type = self._get_data_type(dtype)   
        if data_type == DataType.JSON:
            field_schema = FieldSchema(
                name=schema_name,
                dtype=data_type,
                is_primary=is_primary
            )
        elif data_type == DataType.INT64:
            field_schema = FieldSchema(
                name=schema_name,
                dtype=data_type,
                is_primary=is_primary,
                default=0
            )
        elif data_type == DataType.FLOAT_VECTOR:
            field_schema = FieldSchema(
                name=schema_name,
                dtype=data_type,
                is_primariy=is_primary,
                dim=dim 
            )
        elif data_type == DataType.VARCHAR:
            field_schema = FieldSchema(
                name=schema_name,
                dtype=data_type,
                is_primary=is_primary,
                max_length=max_length 
            )
        return field_schema

    def create_schema(self, field_schema_list, desc, enable_dynamic_field=True):
        schema = CollectionSchema(
            fields=field_schema_list,
            description=desc,
            enable_dynamic_field=enable_dynamic_field
        )
        self.logger.info('Created schema')
        return schema

    def create_index(self, collection, field_name):
        index_params = {
            "metric_type": f"{self.db_config['search_metric']}",
            "index_type": f"{self.db_config['index_type']}",
            "params": {"nlist": f"{self.db_config['index_nlist']}"},
        }   
        collection.create_index(
            field_name=field_name,
            index_params=index_params
        )
        self.logger.info(f'Created index on field: {field_name}')
    
    def create_partition(self, collection, partition_name):
        if not collection.has_partition(partition_name):
            collection.create_partition(partition_name)
            self.logger.info(f'Created partition: {partition_name}')
        else:
            self.logger.warning(f'Partition {partition_name} already exists.')

    def delete_collection(self, collection_name):
        try:
            assert utility.has_collection(collection_name), f'{collection_name}이 존재하지 않습니다.'
            utility.drop_collection(collection_name)
        except:
            pass
    

class DataMilVus(MilVus):   #  args: (DataProcessor)
    '''
    구축된 Milvus DB에 대한 data search, insert 등 작업 수행
    '''
    def __init__(self, db_config):
        super().__init__(db_config)
    
    def get_collection(self, collection_name="news"):
        """
        기본 컬렉션을 가져옵니다. 지정하지 않으면 'news' 컬렉션 사용
        """
        print(f"[DEBUG] Getting collection: {collection_name}")
        try:
            collection = Collection(collection_name)
            print(f"[DEBUG] Collection obtained successfully: {collection_name}")
            return collection
        except Exception as e:
            print(f"[DEBUG] Error getting collection {collection_name}: {str(e)}")
            raise
    
    def delete_data(self, filter, collection_name, filter_type='varchar'):
        '''
        ids: int  - 3  
        expr: str  - "doc_id == 'doc_test'"  
        '''
        collection = Collection(collection_name)
        if filter_type == 'int':        
            collection.delete(ids=[filter])
        elif filter_type == 'varchar':
            collection.delete(expr=filter)

    def insert_data(self, m_data, collection_name, partition_name=None):
        collection = Collection(collection_name)
        collection.insert(m_data, partition_name)
        
    def get_len_data(self, collection):
        print(collection.num_entities)

    def set_search_params(self, query_emb, anns_field='text_emb', expr=None, limit=5, output_fields=None, consistency_level="Strong"):
        self.search_params = {
            "data": [query_emb],
            "anns_field": anns_field, 
            "param": {"metric_type": self.db_config['search_metric'], "params": {"nprobe": 0}, "offset": 0},
            "limit": limit,
            "expr": expr, 
            "output_fields": output_fields,
            "consistency_level": consistency_level
        }
    
    def search_data(self, collection, search_params):
        results = collection.search(**search_params)
        return results

    def get_distance(self, search_result):
        id_list = [] 
        distance_list = [] 
        for idx in range(len(search_result[0])):
            id_list.append(search_result[0][idx].id)
            distance_list.append(search_result[0][idx].distance)
        return id_list, distance_list

    def decode_search_result(self, search_result, include_metadata=False):
        try:
            print(f"[DEBUG] Raw search result type: {type(search_result).__name__}")
            print(f"[DEBUG] Raw search result content: {search_result}")
            
            if include_metadata:
                results = []
                
                # 가장 단순한 직접 추출 방법 먼저 시도
                try:
                    if len(search_result) > 0 and len(search_result[0]) > 0:
                        print(f"[DEBUG] Direct access attempt")
                        for i, hits in enumerate(search_result):
                            for j, hit in enumerate(hits):
                                try:
                                    # Hit 객체에서 직접 속성 추출
                                    result_dict = {}
                                    
                                    # 직접 속성 체크 출력
                                    for attr in dir(hit):
                                        if not attr.startswith('_'):
                                            try:
                                                print(f"[DEBUG] Hit attribute {attr}: {getattr(hit, attr)}")
                                            except:
                                                pass
                                                
                                    # entity 속성 확인 (딕셔너리 형태로 변환)
                                    if hasattr(hit, 'entity') and hit.entity:
                                        if isinstance(hit.entity, dict):
                                            result_dict.update(hit.entity)
                                            print(f"[DEBUG] Added entity dict: {hit.entity}")
                                        elif isinstance(hit.entity, str):
                                            entity_dict = None
                                            try:
                                                # JSON 파싱
                                                entity_dict = json.loads(hit.entity.replace("'", "\""))
                                            except:
                                                try:
                                                    # ast 파싱
                                                    entity_dict = ast.literal_eval(hit.entity)
                                                except:
                                                    pass
                                            
                                            if entity_dict:
                                                result_dict.update(entity_dict)
                                                print(f"[DEBUG] Parsed entity string to dict: {entity_dict}")
                                    
                                    # fields 속성 확인 (직접 활용)
                                    if hasattr(hit, 'fields') and hit.fields:
                                        result_dict.update(hit.fields)
                                        print(f"[DEBUG] Added fields dict: {hit.fields}")
                                    
                                    # id 속성 확인
                                    if hasattr(hit, 'id') and hit.id:
                                        result_dict['id'] = hit.id
                                        print(f"[DEBUG] Added id: {hit.id}")
                                    
                                    # distance 속성 확인
                                    if hasattr(hit, 'distance') and hit.distance is not None:
                                        result_dict['score'] = hit.distance
                                        print(f"[DEBUG] Added score: {hit.distance}")
                                    
                                    # 문자열 표현에서 필요한 정보 추출 (fallback)
                                    if not result_dict:
                                        hit_str = str(hit)
                                        print(f"[DEBUG] Hit string representation: {hit_str}")
                                        
                                        # entity 부분 추출
                                        entity_match = re.search(r"entity:\s*(\{.+\})", hit_str)
                                        if entity_match:
                                            entity_str = entity_match.group(1)
                                            try:
                                                entity_str = entity_str.replace("'", "\"")
                                                entity_str = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', entity_str)
                                                try:
                                                    result_dict.update(json.loads(entity_str))
                                                except:
                                                    entity_str = entity_str.replace("\"", "'")
                                                    result_dict.update(ast.literal_eval(entity_str))
                                                print(f"[DEBUG] Parsed entity from string: {result_dict}")
                                            except Exception as e:
                                                print(f"[DEBUG] Error parsing entity string: {e}")
                                        
                                        # distance 추출
                                        distance_match = re.search(r"distance:\s*([\d.]+)", hit_str)
                                        if distance_match:
                                            result_dict['score'] = float(distance_match.group(1))
                                            print(f"[DEBUG] Extracted score from string: {result_dict['score']}")
                                    
                                    # 결과가 있는 경우만 추가
                                    if result_dict:
                                        results.append(result_dict)
                                        print(f"[DEBUG] Added result: {result_dict}")
                                except Exception as e:
                                    print(f"[DEBUG] Error processing hit directly: {e}")
                except Exception as e:
                    print(f"[DEBUG] Error in direct extraction: {e}")
                
                # 직접 접근 방식으로 결과가 없으면 문자열 파싱 방식 시도
                if not results:
                    print(f"[DEBUG] Attempting string parsing fallback")
                    search_result_str = str(search_result)
                    
                    # 결과 문자열에서 entity 정보 추출
                    entity_matches = re.findall(r"entity:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", search_result_str)
                    for entity_str in entity_matches:
                        try:
                            print(f"[DEBUG] Found entity string: {entity_str}")
                            try:
                                # JSON 파싱 시도
                                entity_str = entity_str.replace("'", "\"")
                                entity_str = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', entity_str)
                                entity_dict = json.loads(entity_str)
                            except:
                                # ast 파싱 시도
                                entity_str = entity_str.replace("\"", "'")
                                entity_dict = ast.literal_eval(entity_str)
                            
                            # distance 추출
                            distance_match = re.search(r"distance:\s*([\d.]+)", search_result_str)
                            if distance_match:
                                entity_dict['score'] = float(distance_match.group(1))
                            
                            results.append(entity_dict)
                            print(f"[DEBUG] Added result from string parsing: {entity_dict}")
                        except Exception as e:
                            print(f"[DEBUG] Error parsing entity string: {e}")
                
                print(f"[DEBUG] Final results count: {len(results)}")
                return results
            else:
                # 텍스트만 반환하는 경우
                texts = []
                
                # 직접 추출 시도
                try:
                    for hits in search_result:
                        for hit in hits:
                            try:
                                # entity에서 text 필드 추출
                                if hasattr(hit, 'entity'):
                                    if isinstance(hit.entity, dict) and 'text' in hit.entity:
                                        texts.append(hit.entity['text'])
                                        print(f"[DEBUG] Extracted text from entity dict: {hit.entity['text']}")
                                
                                # 문자열 표현에서 추출 (fallback)
                                if not texts:
                                    hit_str = str(hit)
                                    entity_match = re.search(r"entity:\s*(\{.+\})", hit_str)
                                    if entity_match:
                                        entity_str = entity_match.group(1)
                                        try:
                                            entity_str = entity_str.replace("'", "\"")
                                            entity_str = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', entity_str)
                                            entity_dict = None
                                            try:
                                                entity_dict = json.loads(entity_str)
                                            except:
                                                entity_str = entity_str.replace("\"", "'")
                                                entity_dict = ast.literal_eval(entity_str)
                                            
                                            if entity_dict and 'text' in entity_dict:
                                                texts.append(entity_dict['text'])
                                                print(f"[DEBUG] Extracted text from string entity: {entity_dict['text']}")
                                        except Exception as e:
                                            print(f"[DEBUG] Error extracting text from entity string: {e}")
                            except Exception as e:
                                print(f"[DEBUG] Error extracting text: {e}")
                except Exception as e:
                    print(f"[DEBUG] Error in direct text extraction: {e}")
                
                # 문자열 파싱 방식 (fallback)
                if not texts:
                    search_result_str = str(search_result)
                    text_matches = re.findall(r"'text':\s*'([^']*)'", search_result_str)
                    for text in text_matches:
                        texts.append(text)
                
                return texts
        except Exception as e:
            print(f"[DEBUG] Search error in decode_search_result: {str(e)}")
            # 문자열 파싱의 최후 수단
            try:
                search_result_str = str(search_result)
                print(f"[DEBUG] Attempting final fallback with raw string: {search_result_str[:200]}...")
                
                # entity 부분 추출 시도
                if include_metadata:
                    entity_matches = re.findall(r"entity:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", search_result_str)
                    results = []
                    for entity_str in entity_matches:
                        try:
                            # 문자열 정제 및 파싱
                            entity_str = entity_str.replace("'", "\"")
                            entity_str = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', entity_str)
                            try:
                                entity_dict = json.loads(entity_str)
                            except:
                                entity_str = entity_str.replace("\"", "'")
                                entity_dict = ast.literal_eval(entity_str)
                            
                            # distance 추출
                            distance_match = re.search(r"distance:\s*([\d.]+)", search_result_str)
                            if distance_match:
                                entity_dict['score'] = float(distance_match.group(1))
                            
                            results.append(entity_dict)
                        except:
                            continue
                    return results
                else:
                    # text 필드만 추출
                    text_matches = re.findall(r"'text':\s*'([^']*)'", search_result_str)
                    return text_matches
            except:
                # 모든 방법 실패 시 빈 리스트 반환
                if include_metadata:
                    return []
                else:
                    return []

    def rerank_data(self, search_result):
        pass 


class MilvusMeta():
    ''' 
    파일이름 - ID Code, 파일이름 - 영문이름 (파티션) 매핑 정보 관리 클래스 
    '''
    def set_rulebook_map(self):
        self.rulebook_id_code = {
            '취업규칙': '00', 
            '윤리규정': '01', 
            '신여비교통비': '02', 
            '경조금지급규정': '03',
            '직무발명보상': '04',
            '투자업무_운영관리': '05',
        }
        self.rulebook_kor_to_eng = {
            '취업규칙': 'employment_rules',
            '윤리규정': 'code_of_ethics',
            '신여비교통비': 'transport_expenses',
            '경조금지급규정': 'extra_expenditure',
            '직무발명보상': 'ei_compensation',
            '투자업무_운영관리': 'io_management'
        }
        self.rulebook_eng_to_kor = {value: key for key, value in self.rulebook_kor_to_eng.items()}


    def rerank_data(self, search_result):
        pass 


class MilvusMeta():
    ''' 
    파일이름 - ID Code, 파일이름 - 영문이름 (파티션) 매핑 정보 관리 클래스 
    '''
    def set_rulebook_map(self):
        self.rulebook_id_code = {
            '취업규칙': '00', 
            '윤리규정': '01', 
            '신여비교통비': '02', 
            '경조금지급규정': '03',
            '직무발명보상': '04',
            '투자업무_운영관리': '05',
        }
        self.rulebook_kor_to_eng = {
            '취업규칙': 'employment_rules',
            '윤리규정': 'code_of_ethics',
            '신여비교통비': 'transport_expenses',
            '경조금지급규정': 'extra_expenditure',
            '직무발명보상': 'ei_compensation',
            '투자업무_운영관리': 'io_management'
        }
        self.rulebook_eng_to_kor = {value: key for key, value in self.rulebook_kor_to_eng.items()}
