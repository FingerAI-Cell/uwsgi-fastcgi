from .milvus import MilvusEnvManager, DataMilVus, MilvusMeta
from .data_p import DataProcessor
from .models import EmbModel
from pymilvus import Collection
import json
import os
from pymilvus import utility
import re
import ast
import time

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
        self.document_metadata = {}  # 문서 메타데이터 저장을 위한 딕셔너리 초기화
        self.loaded_collections = {}  # 로드된 컬렉션 캐싱
        # 캐시 상태 추적 (최대 10개 컬렉션만 캐싱)
        self.max_cached_collections = 10
        self.collection_access_count = {}  # 컬렉션 접근 횟수 추적
    
    def get_collection(self, collection_name):
        """
        컬렉션을 효율적으로 관리합니다.
        - 이미 로드된 컬렉션은 캐시에서 반환
        - 접근 횟수를 추적하여 LRU(Least Recently Used) 방식으로 캐시 관리
        - 메모리 최적화를 위한 캐시 크기 제한
        """
        # 접근 카운트 증가
        if collection_name in self.collection_access_count:
            self.collection_access_count[collection_name] += 1
        else:
            self.collection_access_count[collection_name] = 1
            
        # 이미 로드된 컬렉션이 있는 경우 캐시에서 반환
        if collection_name in self.loaded_collections:
            print(f"[DEBUG] Reusing cached collection: {collection_name}, access count: {self.collection_access_count[collection_name]}")
            collection = self.loaded_collections[collection_name]
            
            # 컬렉션이 로드되어 있는지 확인하고, 필요시 재로드
            try:
                if not collection.is_ready():
                    print(f"[DEBUG] Collection {collection_name} not ready, reloading")
                    collection.load()
            except Exception as e:
                print(f"[WARNING] Error checking collection readiness: {str(e)}, will try to reload")
                try:
                    collection.load()
                except Exception as load_error:
                    print(f"[ERROR] Failed to reload collection: {str(load_error)}")
                    # 캐시에서 제거하고 새로 로드 시도
                    del self.loaded_collections[collection_name]
                    return self._load_new_collection(collection_name)
                
            return collection
        
        # 새 컬렉션 로드
        return self._load_new_collection(collection_name)
    
    def _load_new_collection(self, collection_name):
        """새 컬렉션을 로드하고 캐시에 추가합니다."""
        try:
            print(f"[DEBUG] Loading new collection: {collection_name}")
            
            # 캐시 크기 제한 확인 및 관리
            if len(self.loaded_collections) >= self.max_cached_collections:
                # 가장 적게 접근된 컬렉션 찾기
                least_used = min(
                    self.collection_access_count.items(), 
                    key=lambda x: x[1] if x[0] in self.loaded_collections else float('inf')
                )[0]
                
                if least_used in self.loaded_collections:
                    print(f"[DEBUG] Removing least used collection from cache: {least_used} (access count: {self.collection_access_count[least_used]})")
                    # 캐시에서 제거
                    del self.loaded_collections[least_used]
                    # 메모리 정리 고려 (하지만 릴리스 호출은 하지 않음)
            
            # 새 컬렉션 로드
            collection = Collection(collection_name)
            collection.load()
            
            # 캐시에 추가
            self.loaded_collections[collection_name] = collection
            print(f"[DEBUG] Collection {collection_name} successfully loaded and cached")
            return collection
            
        except Exception as e:
            print(f"[ERROR] Error loading collection {collection_name}: {str(e)}")
            raise

    def parse_json_field(self, field_value):
        """JSON 문자열을 객체로 변환하는 유틸리티 함수"""
        if isinstance(field_value, str):
            try:
                return json.loads(field_value)
            except:
                return field_value
        return field_value
    
    def create_domain(self, domain_name):
        '''
        domain = collection
        '''
        # 각 passage의 고유 식별자
        data_passage_uid = self.vectorenv.create_field_schema('passage_uid', dtype='VARCHAR', is_primary=True, max_length=1024)
        # doc_id는 이제 일반 필드
        data_doc_id = self.vectorenv.create_field_schema('doc_id', dtype='VARCHAR', max_length=1024)
        data_raw_doc_id = self.vectorenv.create_field_schema('raw_doc_id', dtype='VARCHAR', max_length=1024)  # 원본 doc_id 저장용
        data_passage_id = self.vectorenv.create_field_schema('passage_id', dtype='INT64')
        data_domain = self.vectorenv.create_field_schema('domain', dtype='VARCHAR', max_length=32)
        data_title = self.vectorenv.create_field_schema('title', dtype='VARCHAR', max_length=1024)
        data_author = self.vectorenv.create_field_schema('author', dtype='VARCHAR', max_length=128)
        data_text = self.vectorenv.create_field_schema('text', dtype='VARCHAR', max_length=10000)
        data_text_emb = self.vectorenv.create_field_schema('text_emb', dtype='FLOAT_VECTOR', dim=1024)
        data_info = self.vectorenv.create_field_schema('info', dtype='JSON')
        data_tags = self.vectorenv.create_field_schema('tags', dtype='JSON')
        schema_field_list = [data_passage_uid, data_doc_id, data_raw_doc_id, data_passage_id, data_domain, data_title, data_author, data_text, data_text_emb, data_info, data_tags]

        schema = self.vectorenv.create_schema(schema_field_list, 'schema for fai-rag, using fastcgi')
        collection = self.vectorenv.create_collection(domain_name, schema, shards_num=2)
        self.vectorenv.create_index(collection, field_name='text_emb')

    def delete_data(self, domain, doc_id):
        """
        특정 도메인에서 doc_id에 해당하는 모든 passage를 삭제합니다.
        
        Args:
            domain (str): 삭제할 도메인
            doc_id (str): 삭제할 문서 ID
        """
        try:
            print(f"[DEBUG] Deleting all passages for doc_id: {doc_id} in domain: {domain}")
            
            # doc_id가 이미 해시된 값인지 확인
            is_already_hashed = len(doc_id) >= 64 and all(c in '0123456789abcdef' for c in doc_id.lower())
            hashed_doc_id = doc_id if is_already_hashed else self.data_p.hash_text(doc_id, hash_type='blake')
            
            # 삭제할 문서 조건 설정 (작은따옴표 대신 큰따옴표 사용)
            expr = f'doc_id == "{hashed_doc_id}"'
            
            # 삭제 전에 존재하는지 확인
            collection = self.vectordb.get_collection(collection_name=domain)
            print(f"[DEBUG] Checking existence with expression: {expr}")
            
            results = collection.query(
                expr=expr,
                output_fields=["doc_id"],
                limit=1
            )
            
            if not results:
                print(f"[DEBUG] No documents found with doc_id: {doc_id} in domain: {domain}")
                return
            
            # 문서 삭제 실행
            print(f"[DEBUG] Executing delete operation for doc_id: {hashed_doc_id}")
            deleted = collection.delete(expr)
            collection.flush()  # 변경사항을 즉시 적용
            print(f"[DEBUG] Delete operation completed. Deleted {deleted} entries.")
            
            # 문서 메타데이터도 삭제
            doc_metadata_key = f"{domain}:{doc_id}"
            if doc_metadata_key in self.document_metadata:
                del self.document_metadata[doc_metadata_key]
                print(f"[DEBUG] Deleted document metadata for key: {doc_metadata_key}")
            
            print(f"[DEBUG] Successfully deleted all passages for doc_id: {doc_id} in domain: {domain}")
            
        except Exception as e:
            print(f"[ERROR] Failed to delete data: {str(e)}")
            raise

    def insert_data(self, domain, doc_id, title, author, text, info, tags, ignore=True):
        try:
            print(f"[DEBUG] Original text length: {len(text)}")
            
            # doc_id는 이미 해시된 값이고, raw_doc_id는 원본 형식(YYYYMMDD-title-author)
            hashed_doc_id = doc_id
            try:
                date = tags.get('date', '00000000').replace('-','')  # 날짜 없으면 기본값
                raw_doc_id = f"{date}-{title}-{author}"
                if len(raw_doc_id.encode('utf-8')) > 1024:
                    raw_doc_id = raw_doc_id[:200] + "..."  # 길이 제한
            except Exception as e:
                print(f"[WARNING] Error creating raw_doc_id: {str(e)}")
                raw_doc_id = f"unknown_doc_{hashed_doc_id[:8]}"  # 폴백
            print(f"[DEBUG] Hashed doc_id: {hashed_doc_id}, Raw doc_id: {raw_doc_id}")
            
            # 중복 문서 체크
            collection = self.vectordb.get_collection(collection_name=domain)
            expr = f'doc_id == "{hashed_doc_id}"'
            results = collection.query(
                expr=expr,
                output_fields=["doc_id"],
                limit=1
            )
            
            # 중복된 문서가 존재하는 경우
            if results:
                print(f"[DEBUG] Document with doc_id {hashed_doc_id} already exists in domain {domain}")
                if ignore:
                    print(f"[DEBUG] Ignoring insert due to ignore=True")
                    return "skipped"  # 중복으로 인한 건너뛰기 상태 반환
                else:
                    print(f"[DEBUG] Deleting existing document due to ignore=False")
                    # 기존 문서 삭제
                    self.delete_data(domain, doc_id)  # 원본 doc_id 전달
                    print(f"[DEBUG] Successfully deleted existing document")
            
            # 텍스트 청킹 및 삽입
            chunked_texts = self.data_p.chunk_text(text)
            print(f"[DEBUG] Number of chunks: {len(chunked_texts)}")
            
            # info와 tags가 문자열인 경우 파싱
            if isinstance(info, str):
                info = json.loads(info)
            if isinstance(tags, str):
                tags = json.loads(tags)
            
            for i, (chunk, passage_id) in enumerate(chunked_texts):
                print(f"[DEBUG] Processing chunk {i+1}/{len(chunked_texts)}")
                
                # 텍스트 길이 체크 - 새 알고리즘에서는 청크 크기가 최대 약 512바이트로 제한됨
                chunk_bytes = len(chunk.encode('utf-8'))
                if chunk_bytes > 512:  # 스키마 제약에 맞게 정확히 512바이트로 제한
                    print(f"[ERROR] Text chunk too large: {chunk_bytes} bytes exceeds maximum 512 bytes")
                    raise ValueError(f"Text chunk too large: {chunk_bytes} bytes exceeds maximum 512 bytes")
                
                # passage의 고유 식별자 생성
                passage_uid = f"{hashed_doc_id}_{passage_id}"
                
                chunk_emb = self.emb_model.bge_embed_data(chunk)
                data = [
                    {
                        "passage_uid": passage_uid,  # 고유 식별자
                        "doc_id": hashed_doc_id, 
                        "raw_doc_id": raw_doc_id,  # 원본 doc_id 추가
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
                print(f"[DEBUG] Inserting chunk {i+1} with passage_uid: {passage_uid}")
                self.vectordb.insert_data(data, collection_name=domain)
                print(f"[DEBUG] Successfully inserted chunk {i+1}")
            
            return "success"  # 성공적인 삽입 상태 반환
                
        except ValueError as ve:
            print(f"[ERROR] Validation error: {str(ve)}")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to insert data: {str(e)}")
            raise
    
    def retrieve_data(self, query, top_k, filter_conditions=None):
        """
        텍스트 검색과 다양한 필터링 조건을 조합하여 검색을 수행합니다.
        
        Args:
            query (str): 검색할 텍스트 (필수)
            top_k (int): 반환할 결과 개수
            filter_conditions (dict): 필터링 조건
                - domain: 단일 도메인 검색용
                - domains: 복수 도메인 검색용 (리스트)
                - date_range: {'start': 'YYYYMMDD', 'end': 'YYYYMMDD'}
                  (날짜는 YYYYMMDD 형식의 문자열, 예: '20220804')
                - title: 제목 검색어
                - info: info 필드 내 검색 조건
                - tags: tags 필드 내 검색 조건
        """
        start_time = time.time()
        
        print(f"[DEBUG] retrieve_data: query={query}, top_k={top_k}, filter_conditions={filter_conditions}")
        cleansed_text = self.data_p.cleanse_text(query)
        print(f"[DEBUG] cleansed_text={cleansed_text}")
        
        cleanse_time = time.time()
        print(f"[TIMING] 텍스트 전처리 완료: {(cleanse_time - start_time):.4f}초")
        
        # 기본 출력 필드 설정
        output_fields = ["doc_id", "raw_doc_id", "passage_id", "domain", "title", "author", "text", "info", "tags"]
        
        # 검색 표현식 구성
        expr_parts = []
        
        # 도메인 필터
        domain = None
        if filter_conditions:
            if 'domain' in filter_conditions:
                domain = filter_conditions['domain']
            elif 'domains' in filter_conditions:
                # domains가 있으면 단일 도메인 검색에서는 무시 (app.py에서 처리)
                return []
        
        if not domain:
            # 도메인이 지정되지 않은 경우 기본 도메인 사용
            domain = "news"  # 기본 도메인 설정
        
        print(f"[DEBUG] Using domain: {domain}")
        
        # 컬렉션이 있는지 확인 (신속히 처리)
        collection_check_start = time.time()
        available_collections = utility.list_collections()
        
        if domain not in available_collections:
            print(f"[DEBUG] Collection {domain} not found")
            return []
        
        collection_check_end = time.time()
        print(f"[TIMING] 컬렉션 확인 완료: {(collection_check_end - collection_check_start):.4f}초")
            
        # 벡터 검색 수행 - 쿼리 임베딩을 먼저 계산 (컬렉션 로드와 병렬로 처리 가능)
        embed_start = time.time()
        query_emb = self.emb_model.bge_embed_data(cleansed_text)
        print(f"[DEBUG] Generated query embedding, length: {len(query_emb)}")
        
        embed_end = time.time()
        print(f"[TIMING] 임베딩 생성 완료: {(embed_end - embed_start):.4f}초")
        
        # 캐싱된 컬렉션 가져오기 (임베딩 생성 후에 처리)
        collection_load_start = time.time()
        try:
            collection = self.get_collection(domain)
            print(f"[DEBUG] Collection {domain} loaded, num_entities: {collection.num_entities}")
        except Exception as e:
            print(f"[ERROR] Failed to load collection {domain}: {str(e)}")
            return []
        
        collection_load_end = time.time()
        print(f"[TIMING] 컬렉션 로드 완료: {(collection_load_end - collection_load_start):.4f}초")
        
        # 검색 표현식 구성
        expr_build_start = time.time()
        
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
        print(f"[DEBUG] Final search expression: {expr}")
        
        expr_build_end = time.time()
        print(f"[TIMING] 검색 표현식 구성 완료: {(expr_build_end - expr_build_start):.4f}초")
        
        # 검색 파라미터 설정
        search_params_start = time.time()
        self.vectordb.set_search_params(
            query_emb, 
            limit=top_k, 
            output_fields=output_fields,
            expr=expr
        )
        print(f"[DEBUG] Search params: {self.vectordb.search_params}")
        
        search_params_end = time.time()
        print(f"[TIMING] 검색 파라미터 설정 완료: {(search_params_end - search_params_start):.4f}초")
        
        # 실제 검색 실행
        try:
            search_exec_start = time.time()
            search_result = self.vectordb.search_data(collection, self.vectordb.search_params)
            print(f"[DEBUG] Search result: {search_result}")
            
            search_exec_end = time.time()
            print(f"[TIMING] 검색 실행 완료: {(search_exec_end - search_exec_start):.4f}초")
            
            decode_start = time.time()
            results = self.vectordb.decode_search_result(search_result, include_metadata=True)
            
            # info와 tags JSON 문자열을 객체로 변환 (개선된 방식)
            for result in results:
                result['info'] = self.parse_json_field(result.get('info'))
                result['tags'] = self.parse_json_field(result.get('tags'))
            
            decode_end = time.time()
            print(f"[TIMING] 결과 디코딩 완료: {(decode_end - decode_start):.4f}초")
            
            print(f"[DEBUG] Decoded results: {results}")
            
            end_time = time.time()
            total_time = end_time - start_time
            print(f"[TIMING] 전체 검색 처리 완료: 총 {total_time:.4f}초 소요")
            
            return results
            
        except Exception as e:
            error_time = time.time()
            total_time = error_time - start_time
            print(f"[TIMING] 검색 오류 발생: {total_time:.4f}초 소요, 오류: {str(e)}")
            print(f"[DEBUG] Search error: {str(e)}")
            return []

    def get_document_passages(self, doc_id, domains):
        """
        특정 문서의 모든 패시지를 가져옴
        - doc_id: 문서 ID
        - domains: 검색할 도메인 리스트
        - 도메인별 문서 정보와 패시지 리스트 반환
        """
        if not doc_id:
            print("[WARNING] Empty doc_id")
            return []
            
        if not domains:
            print("[WARNING] No domains specified")
            return []
            
        # doc_id 해시 처리
        is_already_hashed = len(doc_id) >= 64 and all(c in '0123456789abcdef' for c in doc_id.lower())
        hashed_doc_id = doc_id if is_already_hashed else self.data_p.hash_text(doc_id, hash_type='blake')
        print(f"[DEBUG] Original doc_id: {doc_id}, Hashed doc_id: {hashed_doc_id}")
        
        domain_results = {}
        for domain in domains:
            try:
                collection = self.vectordb.get_collection(domain)
                print(f"[DEBUG] Collection obtained successfully: {domain}")
                
                try:
                    expr = f'doc_id == "{hashed_doc_id}"'
                    print(f"[DEBUG] Searching collection {domain} with expr: {expr}")
                    
                    results = collection.query(
                        expr=expr,
                        output_fields=["doc_id", "raw_doc_id", "passage_id", "domain", "title", "author", "text", "info", "tags"]
                    )
                    
                    if results:
                        print(f"[DEBUG] Found {len(results)} passages in {domain}")
                        
                        # 첫 번째 passage에서 해당 도메인의 문서 정보 추출
                        first_passage = results[0]
                        
                        # info와 tags JSON 파싱
                        info = first_passage["info"]
                        tags = first_passage["tags"]
                        if isinstance(info, str):
                            try:
                                info = json.loads(info)
                            except:
                                pass
                        if isinstance(tags, str):
                            try:
                                tags = json.loads(tags)
                            except:
                                pass
                        
                        domain_result = {
                            "doc_id": hashed_doc_id,
                            "raw_doc_id": first_passage.get("raw_doc_id", doc_id),
                            "domain": domain,
                            "title": first_passage["title"],
                            "author": first_passage["author"],
                            "info": info,
                            "tags": tags,
                            "passages": []
                        }
                        
                        # passage 정보 추출하여 정렬 후 추가
                        for passage in sorted(results, key=lambda x: int(x["passage_id"])):
                            domain_result["passages"].append({
                                "passage_id": passage["passage_id"],
                                "text": passage["text"],
                                "position": passage["passage_id"]
                            })
                            
                        domain_results[domain] = domain_result
                        
                except Exception as e:
                    print(f"[ERROR] Failed to query collection {domain}: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"[ERROR] Failed to get collection {domain}: {str(e)}")
                continue
        
        if not domain_results:
            print(f"[DEBUG] No passages found for doc_id: {doc_id} in specified domains")
            return None
            
        result = {
            "doc_id": hashed_doc_id,
            "raw_doc_id": doc_id,
            "domain_results": domain_results
        }
        
        print(f"[DEBUG] Retrieved results from {len(domain_results)} domains")
        return result

    def get_specific_passage(self, doc_id, passage_id, domains):
        """
        doc_id와 passage_id로 특정 패시지를 조회합니다.
        지정된 도메인에서만 검색을 수행합니다.
        """
        try:
            print(f"[DEBUG] get_specific_passage called with doc_id: {doc_id}, passage_id: {passage_id}")
            
            if not domains:
                print("[WARNING] No domains specified")
                return None
            
            # 이미 해시된 ID인지 확인
            is_already_hashed = len(doc_id) >= 64 and all(c in '0123456789abcdef' for c in doc_id.lower())
            
            if is_already_hashed:
                print(f"[DEBUG] doc_id appears to be already hashed, using as-is")
                hashed_doc_id = doc_id
            else:
                hashed_doc_id = self.data_p.hash_text(doc_id, hash_type='blake')
                print(f"[DEBUG] Original doc_id hashed to: {hashed_doc_id}")
            
            # passage_id가 문자열인 경우 처리
            try:
                if isinstance(passage_id, str):
                    if passage_id.startswith('p'):
                        passage_num = int(passage_id[1:])
                    else:
                        passage_num = int(passage_id)
                else:
                    passage_num = int(passage_id)
                print(f"[DEBUG] Converted passage_id to number: {passage_num}")
            except (ValueError, TypeError) as e:
                print(f"[DEBUG] Error converting passage_id: {e}")
                return None
            
            # 지정된 도메인에서만 검색
            for domain in domains:
                try:
                    print(f"[DEBUG] Searching in collection: {domain}")
                    collection = self.vectordb.get_collection(collection_name=domain)
                    print(f"[DEBUG] Collection obtained: {collection.name}, entities: {collection.num_entities}")
                    
                    # doc_id와 passage_id로 특정 패시지 조회
                    expr = f"doc_id == '{hashed_doc_id}' && passage_id == {passage_num}"
                    print(f"[DEBUG] Query expression: {expr}")
                    
                    try:
                        results = collection.query(
                            expr=expr,
                            output_fields=["doc_id", "raw_doc_id", "passage_id", "domain", "title", "author", "text", "info", "tags"]
                        )
                        print(f"[DEBUG] Query results count: {len(results) if results else 0}")
                        
                        if results:
                            passage = results[0]
                            
                            # info와 tags JSON 파싱
                            info = passage["info"]
                            tags = passage["tags"]
                            if isinstance(info, str):
                                try:
                                    info = json.loads(info)
                                except:
                                    pass
                            if isinstance(tags, str):
                                try:
                                    tags = json.loads(tags)
                                except:
                                    pass
                            
                            result = {
                                "doc_id": doc_id,
                                "raw_doc_id": passage.get("raw_doc_id", doc_id),
                                "passage_id": passage['passage_id'],
                                "text": passage["text"],
                                "position": passage["passage_id"],
                                "metadata": {
                                    "domain": passage["domain"],
                                    "title": passage["title"],
                                    "info": info,
                                    "tags": tags
                                }
                            }
                            print(f"[DEBUG] Found passage in collection {domain}")
                            return result
                            
                    except Exception as e:
                        print(f"[DEBUG] Error querying collection {domain}: {str(e)}")
                        continue
                        
                except Exception as e:
                    print(f"[DEBUG] Error accessing collection {domain}: {str(e)}")
                    continue
            
            print(f"[DEBUG] Passage not found in specified domains")
            return None
            
        except Exception as e:
            print(f"[DEBUG] Unhandled exception in get_specific_passage: {str(e)}")
            return None

    def raw_insert_data(self, domain, doc_id, passage_id, title, author, text, info={}, tags={}, ignore=True):
        '''
        텍스트를 분할하지 않고 그대로 저장하는 메소드
        
        Args:
            domain (str): 컬렉션 이름
            doc_id (str): 문서 ID (사용자 지정)
            passage_id (int): 패시지 ID (사용자 지정)
            title (str): 문서 제목
            author (str): 작성자
            text (str): 문서 본문
            info (dict): 추가 정보
            tags (dict): 태그 정보
            ignore (bool): 중복 문서 처리 방식
                         - True: 중복 시 건너뜀
                         - False: 중복 시 삭제 후 재생성
            
        Returns:
            str: "success" (새로 생성) 또는 "skipped" (건너뜀) 또는 "updated" (업데이트)
        '''
        try:
            # 도메인이 없으면 생성
            if domain not in self.vectorenv.get_list_collection():
                print(f"[DEBUG] Creating new collection: {domain}")
                self.create_domain(domain)
                print(f"[DEBUG] Collection created successfully")
            
            # doc_id는 사용자 입력값을 raw_doc_id로 사용하고 해시
            raw_doc_id = doc_id  # 사용자 입력값을 원본으로 저장
            if len(raw_doc_id.encode('utf-8')) > 1024:
                raise ValueError("doc_id is too long (max 1024 bytes)")
            
            hashed_doc_id = self.data_p.hash_text(doc_id, hash_type='blake')
            print(f"[DEBUG] Raw doc_id: {raw_doc_id}, Hashed doc_id: {hashed_doc_id}")
            
            # 컬렉션 로드
            collection = Collection(domain)
            collection.load()
            print(f"[DEBUG] Collection loaded: {collection.name}, num_entities: {collection.num_entities}")
            
            # passage_uid 생성 (해시된 doc_id 사용)
            passage_uid = f"{hashed_doc_id}-p{passage_id}"
            
            # 중복 체크 (큰따옴표 사용)
            expr = f'doc_id == "{hashed_doc_id}" && passage_id == {passage_id}'
            res = collection.query(expr)
            
            if res:
                if ignore:
                    return "skipped"
                else:
                    # 기존 문서 삭제 후 재생성
                    collection.delete(expr)
                    collection.flush()
                    is_update = True
            else:
                is_update = False
            
            # 텍스트 임베딩
            text_emb = self.emb_model.bge_embed_data(text)
            print(f"[DEBUG] Generated embedding, length: {len(text_emb)}")
            
            # info와 tags가 문자열인 경우 파싱
            if isinstance(info, str):
                info = json.loads(info)
            if isinstance(tags, str):
                tags = json.loads(tags)
            
            # 데이터 삽입 (해시된 doc_id 사용)
            data = [
                {
                    "passage_uid": passage_uid,
                    "doc_id": hashed_doc_id,
                    "raw_doc_id": raw_doc_id,  # 원본 doc_id 추가
                    "passage_id": passage_id,
                    "domain": domain,
                    "title": title,
                    "author": author,
                    "text": text,
                    "text_emb": text_emb,
                    "info": info,
                    "tags": tags
                }
            ]
            
            print(f"[DEBUG] Inserting data with passage_uid: {passage_uid}")
            self.vectordb.insert_data(data, collection_name=domain)
            collection.flush()
            print(f"[DEBUG] Data inserted and flushed successfully")
            
            # 인덱스가 있는지 확인하고 없으면 생성
            if not collection.has_index():
                print(f"[DEBUG] Creating index for collection")
                self.vectorenv.create_index(collection, field_name='text_emb')
                print(f"[DEBUG] Index created successfully")
            
            # 컬렉션 다시 로드하여 인덱스 적용
            collection.load()
            print(f"[DEBUG] Collection reloaded with index")
            
            return "updated" if is_update else "success"
            
        except Exception as e:
            print(f"Error in raw_insert_data: {str(e)}")
            raise