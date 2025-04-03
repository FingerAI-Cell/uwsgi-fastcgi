from .milvus import MilvusEnvManager, DataMilVus, MilvusMeta
from .data_p import DataProcessor
from .models import EmbModel
from pymilvus import Collection
import json
import os
from pymilvus import utility
import re
import ast

class EnvManager():
    def __init__(self, args):
        self.args = args
        self.set_config()
        self.cohere_api = os.getenv('COHERE_API_KEY')   
        self.db_config['ip_addr'] = self.args['ip_addr']
        
    def set_config(self):
        with open(os.path.join(self.args['config_path'], self.args['db_config'])) as f:
            self.db_config = json.load(f)
        with open(os.path.join(self.args['config_path'], self.args['llm_config'])) as f:
            self.llm_config = json.load(f)
    
    def set_processors(self):
        self.data_p = DataProcessor()
        
    def set_vectordb(self):
        self.milvus_db = MilvusEnvManager(self.db_config)
        data_milvus = DataMilVus(self.db_config)
        meta_milvus = MilvusMeta()
        return data_milvus, meta_milvus 

    def set_emb_model(self):
        emb_model = EmbModel(self.llm_config)
        emb_model.set_emb_model(model_type='bge')
        emb_model.set_embbeding_config()
        return emb_model         


class InteractManager:
    def __init__(self, data_p=None, vectorenv=None, vectordb=None, emb_model=None, response_model=None, logger=None):
        '''
        vectordb = MilvusData - insert data, set search params, search data 
        '''
        self.data_p = data_p
        self.vectorenv = vectorenv
        self.vectordb = vectordb 
        self.emb_model = emb_model 
        self.response_model = response_model 
        self.logger = logger 
    
    def create_domain(self, domain_name):
        '''
        domain = collection
        '''
        data_doc_id = self.vectorenv.create_field_schema('doc_id', dtype='VARCHAR', is_primary=True, max_length=1024)
        data_passage_id = self.vectorenv.create_field_schema('passage_id', dtype='INT64')
        data_domain = self.vectorenv.create_field_schema('domain', dtype='VARCHAR', max_length=32)
        data_title = self.vectorenv.create_field_schema('title', dtype='VARCHAR', max_length=128)
        data_author = self.vectorenv.create_field_schema('author', dtype='VARCHAR', max_length=128)  # 작성자 필드 추가
        data_text = self.vectorenv.create_field_schema('text', dtype='VARCHAR', max_length=512)   # 500B (500글자 단위로 문서 분할)
        data_text_emb = self.vectorenv.create_field_schema('text_emb', dtype='FLOAT_VECTOR', dim=1024)
        data_info = self.vectorenv.create_field_schema('info', dtype='JSON')
        data_tags = self.vectorenv.create_field_schema('tags', dtype='JSON')
        schema_field_list = [data_doc_id, data_passage_id, data_domain, data_title, data_author, data_text, data_text_emb, data_info, data_tags]

        schema = self.vectorenv.create_schema(schema_field_list, 'schema for fai-rag, using fastcgi')
        collection = self.vectorenv.create_collection(domain_name, schema, shards_num=2)
        self.vectorenv.create_index(collection, field_name='text_emb')   # doc_id 필드에 index 생성 

    def delete_data(self, domain, doc_id):
        hashed_doc_id = self.data_p.hash_text(doc_id, hash_type='blake')
        data_to_delete = f"doc_id == {hashed_doc_id}"  
        self.vectordb.delete_data(filter=data_to_delete, collection_name=domain)

    def insert_data(self, domain, doc_id, title, author, text, info, tags):
        hashed_doc_id = self.data_p.hash_text(doc_id, hash_type='blake')
        chunked_texts = self.data_p.chunk_text(text)
        for chunk in chunked_texts:
            chunk, passage_id = chunk[0], chunk[1]
            chunk_emb = self.emb_model.bge_embed_data(chunk)
            data = [
                {
                    "doc_id": hashed_doc_id, 
                    "passage_id": passage_id, 
                    "domain": domain, 
                    "title": title, 
                    "author": author,
                    "text": chunk, 
                    "text_emb": chunk_emb, 
                    "info": info, 
                    "tags": tags
                }
            ]        
            self.vectordb.insert_data(data, collection_name=domain)
    
    def retrieve_data(self, query, top_k, filter_conditions=None):
        """
        텍스트 검색과 다양한 필터링 조건을 조합하여 검색을 수행합니다.
        
        Args:
            query (str): 검색할 텍스트 (필수)
            top_k (int): 반환할 결과 개수
            filter_conditions (dict): 필터링 조건
                - domain: 도메인 필터
                - date_range: {'start': 'YYYYMMDD', 'end': 'YYYYMMDD'}
                  (날짜는 YYYYMMDD 형식의 문자열, 예: '20220804')
                - title: 제목 검색어
                - info: info 필드 내 검색 조건
                - tags: tags 필드 내 검색 조건
        """
        print(f"[DEBUG] retrieve_data: query={query}, top_k={top_k}, filter_conditions={filter_conditions}")
        cleansed_text = self.data_p.cleanse_text(query)
        print(f"[DEBUG] cleansed_text={cleansed_text}")
        
        # 기본 출력 필드 설정
        output_fields = ["doc_id", "passage_id", "domain", "title", "text", "info", "tags"]
        
        # 검색 표현식 구성
        expr_parts = []
        
        # 도메인 필터
        domain = filter_conditions.get('domain') if filter_conditions else None
        
        if not domain:
            # 도메인이 지정되지 않은 경우 기본 도메인 사용
            domain = "news"  # 기본 도메인 설정
        
        print(f"[DEBUG] Using domain: {domain}")
        
        # 컬렉션이 있는지 확인하고 없으면 생성
        collections = utility.list_collections()
        print(f"[DEBUG] Available collections: {collections}")
        
        if domain not in collections:
            print(f"[DEBUG] Collection {domain} not found")
            # 도메인에 해당하는 컬렉션이 없으면 오류 반환
            return []
            
        # 컬렉션을 로드하고 사용
        collection = Collection(domain)
        collection.load()
        print(f"[DEBUG] Collection {domain} loaded, num_entities: {collection.num_entities}")
        
        if domain:
            expr_parts.append(f"domain == '{domain}'")
        
        # 날짜 범위 필터 (YYYYMMDD 형식 문자열 비교)
        if filter_conditions and 'date_range' in filter_conditions:
            date_range = filter_conditions['date_range']
            # 날짜는 문자열 비교를 사용하되, YYYYMMDD 형식이므로 직접 비교 가능
            if date_range.get('start'):
                expr_parts.append(f"tags['date'] >= '{date_range['start']}'")
            if date_range.get('end'):
                expr_parts.append(f"tags['date'] <= '{date_range['end']}'")
        
        # 제목 필터
        if filter_conditions and 'title' in filter_conditions:
            title_query = filter_conditions['title'].replace("'", "''")  # SQL injection 방지
            expr_parts.append(f"title like '%{title_query}%'")
        
        # info 필드 필터
        if filter_conditions and 'info' in filter_conditions:
            for key, value in filter_conditions['info'].items():
                # SQL injection 방지를 위한 이스케이프 처리
                key = key.replace("'", "''")
                if isinstance(value, str):
                    value = value.replace("'", "''")
                    expr_parts.append(f"info['{key}'] == '{value}'")
                else:
                    expr_parts.append(f"info['{key}'] == {value}")
        
        # tags 필드 필터
        if filter_conditions and 'tags' in filter_conditions:
            for key, value in filter_conditions['tags'].items():
                if key != 'date':  # 날짜는 이미 처리됨
                    # SQL injection 방지를 위한 이스케이프 처리
                    key = key.replace("'", "''")
                    if isinstance(value, str):
                        value = value.replace("'", "''")
                        expr_parts.append(f"tags['{key}'] == '{value}'")
                    else:
                        expr_parts.append(f"tags['{key}'] == {value}")
        
        # 최종 검색 표현식 구성
        expr = " && ".join(expr_parts) if expr_parts else None
        
        # 벡터 검색 수행
        query_emb = self.emb_model.bge_embed_data(cleansed_text)
        print(f"[DEBUG] Generated query embedding, length: {len(query_emb)}")
        
        self.vectordb.set_search_params(
            query_emb, 
            limit=top_k, 
            output_fields=output_fields,
            expr=expr
        )
        print(f"[DEBUG] Search params: {self.vectordb.search_params}")
        
        try:
            search_result = self.vectordb.search_data(collection, self.vectordb.search_params)
            print(f"[DEBUG] Search result: {search_result}")
            results = self.vectordb.decode_search_result(search_result, include_metadata=True)
            print(f"[DEBUG] Decoded results: {results}")
        except Exception as e:
            print(f"[DEBUG] Search error: {str(e)}")
            return []
        
        # 검색 결과 포맷팅
        formatted_results = []
        for result in results:
            print(f"[DEBUG] Formatting result: {result}")
            
            # 기본 결과 구조
            formatted_result = {
                "doc_id": None,
                "passage_id": None,
                "domain": None,
                "title": None,
                "text": None,
                "info": None,
                "tags": None
            }
            
            # fields 속성이 있으면 활용
            if hasattr(result, 'fields') and result.fields:
                print(f"[DEBUG] Using fields attribute: {result.fields}")
                formatted_result.update(result.fields)
            elif 'fields' in result:
                print(f"[DEBUG] Using fields from dict: {result['fields']}")
                formatted_result.update(result['fields'])
            else:
                # 일반 딕셔너리에서 직접 키 추출
                for key in ['doc_id', 'passage_id', 'domain', 'title', 'text', 'info', 'tags']:
                    if key in result:
                        formatted_result[key] = result[key]
            
            # Hit 속성에서 텍스트 추출
            hit_str = str(result)
            entity_match = re.search(r"entity:\s*(\{.+\})", hit_str)
            if entity_match:
                try:
                    entity_str = entity_match.group(1)
                    entity_str = entity_str.replace("'", "\"")
                    entity_str = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', entity_str)
                    try:
                        entity_dict = json.loads(entity_str)
                        print(f"[DEBUG] Using hit string entity: {entity_dict}")
                        formatted_result.update(entity_dict)
                    except:
                        try:
                            entity_str = entity_str.replace("\"", "'")
                            entity_dict = ast.literal_eval(entity_str)
                            print(f"[DEBUG] Using hit string entity with ast: {entity_dict}")
                            formatted_result.update(entity_dict)
                        except Exception as e:
                            print(f"[DEBUG] Error parsing entity string: {e}")
                except Exception as e:
                    print(f"[DEBUG] Error extracting entity: {e}")
            
            # score 속성 추가
            if 'score' in result:
                formatted_result["score"] = float(result['score'])
            elif hasattr(result, 'score'):
                formatted_result["score"] = float(result.score)
            
            print(f"[DEBUG] Final formatted result: {formatted_result}")
            formatted_results.append(formatted_result)
        
        return formatted_results

    def get_document_passages(self, doc_id):
        """
        doc_id로 문서의 모든 패시지를 조회합니다.
        """
        try:
            print(f"[DEBUG] get_document_passages called with doc_id: {doc_id}")
            
            # doc_id가 None이거나 빈 문자열인 경우 처리
            if not doc_id:
                print("[DEBUG] doc_id is empty")
                return None
            
            # 이미 해시된 ID인지 확인 (64자 이상의 16진수 문자열)
            is_already_hashed = len(doc_id) >= 64 and all(c in '0123456789abcdef' for c in doc_id.lower())
            
            if is_already_hashed:
                print(f"[DEBUG] doc_id appears to be already hashed, using as-is")
                hashed_doc_id = doc_id
            else:
                hashed_doc_id = self.data_p.hash_text(doc_id, hash_type='blake')
                print(f"[DEBUG] Original doc_id hashed to: {hashed_doc_id}")
                
            print(f"[DEBUG] Using hashed_doc_id: {hashed_doc_id}")
            
            try:
                # 도메인 이름을 지정해야 함 (기본값으로 'news' 사용)
                collection = self.vectordb.get_collection(collection_name="news")
                print(f"[DEBUG] Collection obtained: {collection.name}, entities: {collection.num_entities}")
            except Exception as e:
                print(f"[DEBUG] Error getting collection: {str(e)}")
                return None
            
            # doc_id로 문서 조회
            expr = f"doc_id == '{hashed_doc_id}'"
            print(f"[DEBUG] Query expression: {expr}")
            
            try:
                results = collection.query(
                    expr=expr,
                    output_fields=["doc_id", "passage_id", "domain", "title", "text", "info", "tags"]
                )
                print(f"[DEBUG] Query results count: {len(results) if results else 0}")
            except Exception as e:
                print(f"[DEBUG] Error querying collection: {str(e)}")
                return None
            
            if not results:
                print(f"[DEBUG] No results found for doc_id: {doc_id}, hashed: {hashed_doc_id}")
                return None
                
            # 결과 정리
            doc_info = {
                "doc_id": doc_id,
                "domain": results[0]["domain"],
                "title": results[0]["title"],
                "info": results[0]["info"],
                "tags": results[0]["tags"],
                "passages": []
            }
            
            # 패시지 정렬 및 추가
            try:
                sorted_passages = sorted(results, key=lambda x: x["passage_id"])
                for passage in sorted_passages:
                    doc_info["passages"].append({
                        "passage_id": passage['passage_id'],
                        "text": passage["text"],
                        "position": passage["passage_id"]
                    })
                print(f"[DEBUG] Processed {len(doc_info['passages'])} passages")
            except Exception as e:
                print(f"[DEBUG] Error processing passages: {str(e)}")
                # 에러가 있더라도 기본 정보는 반환
                
            return doc_info
            
        except Exception as e:
            print(f"[DEBUG] Unhandled exception in get_document_passages: {str(e)}")
            return None

    def get_specific_passage(self, doc_id, passage_id):
        """
        doc_id와 passage_id로 특정 패시지를 조회합니다.
        """
        try:
            print(f"[DEBUG] get_specific_passage called with doc_id: {doc_id}, passage_id: {passage_id}")
            
            # 이미 해시된 ID인지 확인 (64자 이상의 16진수 문자열)
            is_already_hashed = len(doc_id) >= 64 and all(c in '0123456789abcdef' for c in doc_id.lower())
            
            if is_already_hashed:
                print(f"[DEBUG] doc_id appears to be already hashed, using as-is")
                hashed_doc_id = doc_id
            else:
                hashed_doc_id = self.data_p.hash_text(doc_id, hash_type='blake')
                print(f"[DEBUG] Original doc_id hashed to: {hashed_doc_id}")
                
            print(f"[DEBUG] Using hashed_doc_id: {hashed_doc_id}")
            
            # passage_id가 문자열인 경우 처리
            try:
                if isinstance(passage_id, str):
                    # 'p'로 시작하는 경우 'p' 제거
                    if passage_id.startswith('p'):
                        passage_num = int(passage_id[1:])
                    else:
                        # 숫자 문자열인 경우 그대로 변환
                        passage_num = int(passage_id)
                else:
                    # 이미 숫자인 경우 그대로 사용
                    passage_num = int(passage_id)
                print(f"[DEBUG] Converted passage_id to number: {passage_num}")
            except (ValueError, TypeError) as e:
                print(f"[DEBUG] Error converting passage_id: {e}")
                return None
            
            try:
                # 도메인 이름을 지정해야 함 (기본값으로 'news' 사용)
                collection = self.vectordb.get_collection(collection_name="news")
                print(f"[DEBUG] Collection obtained: {collection.name}, entities: {collection.num_entities}")
            except Exception as e:
                print(f"[DEBUG] Error getting collection: {str(e)}")
                return None
            
            # doc_id와 passage_id로 특정 패시지 조회
            expr = f"doc_id == '{hashed_doc_id}' && passage_id == {passage_num}"
            print(f"[DEBUG] Query expression: {expr}")
            
            try:
                results = collection.query(
                    expr=expr,
                    output_fields=["doc_id", "passage_id", "domain", "title", "text", "info", "tags"]
                )
                print(f"[DEBUG] Query results count: {len(results) if results else 0}")
            except Exception as e:
                print(f"[DEBUG] Error querying collection: {str(e)}")
                return None
            
            if not results:
                print(f"[DEBUG] No results found for query: {expr}")
                return None
                
            passage = results[0]
            result = {
                "doc_id": doc_id,
                "passage_id": passage['passage_id'],
                "text": passage["text"],
                "position": passage["passage_id"],
                "metadata": {
                    "domain": passage["domain"],
                    "title": passage["title"],
                    "info": passage["info"],
                    "tags": passage["tags"]
                }
            }
            print(f"[DEBUG] Returning result for passage: {result['passage_id']}")
            return result
            
        except Exception as e:
            print(f"[DEBUG] Unhandled exception in get_specific_passage: {str(e)}")
            return None