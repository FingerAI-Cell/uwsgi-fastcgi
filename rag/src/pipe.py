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
import concurrent.futures
import threading
import logging
import unicodedata
import traceback
import uuid
from collections import defaultdict
from queue import Queue
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# 전역 GPU 세마포어 설정 - 모든 임베딩 처리에서 공유
GPU_WORKERS = int(os.getenv('MAX_GPU_WORKERS', '50'))
_GPU_SEMAPHORE = threading.Semaphore(GPU_WORKERS)
GPU_TASKS_ACTIVE = 0
GPU_TASK_LOCK = threading.Lock()

class EnvManager():
    def __init__(self, args):
        self.args = args
        self.set_config()
        self.set_processors()
        self.set_vectordb()
        self.set_emb_model()
        
        # 로그 디렉토리 확인 및 생성
        self.log_dir = "/var/log/rag" if os.path.exists("/var/log/rag") else "../logs"
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"EnvManager - 로그 디렉토리: {self.log_dir}")
        
    def set_config(self):
        # 설정 파일에서 기본 설정 로드
        with open(os.path.join(self.args['config_path'], self.args['db_config'])) as f:
            self.db_config = json.load(f)
        with open(os.path.join(self.args['config_path'], self.args['llm_config'])) as f:
            self.llm_config = json.load(f)
            
        # args에서 전달된 값들을 db_config에 추가
        # 특히 환경 변수에서 가져온 ip_addr을 db_config에 추가
        if 'ip_addr' in self.args and self.args['ip_addr']:
            self.db_config['ip_addr'] = self.args['ip_addr']
            print(f"[CONFIG] Using ip_addr from environment: {self.args['ip_addr']}")
        else:
            # ip_addr이 없는 경우 기본값 설정
            self.db_config.setdefault('ip_addr', 'milvus-standalone')
            print(f"[CONFIG] Using default ip_addr: {self.db_config['ip_addr']}")
    
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
    # 클래스 변수로 세마포어 공유
    _gpu_semaphore = _GPU_SEMAPHORE
    _task_lock = GPU_TASK_LOCK
    
    # 글로벌 배치 큐 관련 변수 추가
    global_batch_queue = {}  # 도메인별 글로벌 배치 큐 {domain: [chunks...]}
    global_batch_lock = threading.Lock()  # 글로벌 배치 큐 접근을 위한 락
    batch_worker_thread = None  # 배치 처리 워커 스레드
    batch_worker_running = False  # 워커 스레드 실행 상태
    # 로깅 용도로만 사용
    last_batch_time = {}  # 도메인별 마지막 배치 추가 시간 {domain: timestamp}
    
    @classmethod
    def get_gpu_semaphore(cls):
        """모든 인스턴스에서 공유하는 GPU 세마포어를 반환합니다."""
        # 세마포어가 초기화되지 않았으면 초기화
        if cls._gpu_semaphore is None:
            cls._gpu_semaphore = threading.BoundedSemaphore(int(os.getenv('MAX_GPU_WORKERS', '50')))
        return cls._gpu_semaphore
    
    @classmethod
    def get_gpu_semaphore_value(cls):
        """현재 GPU 세마포어 값(사용 가능한 슬롯 수)을 반환합니다."""
        sem = cls.get_gpu_semaphore()
        if hasattr(sem, '_value'):
            return sem._value
        return 'unknown'
    
    @classmethod
    def get_max_workers(cls):
        """최대 GPU 작업 수 설정을 반환합니다."""
        return int(os.getenv('MAX_GPU_WORKERS', '50'))
    
    @classmethod
    def get_active_workers(cls):
        """현재 활성화된 GPU 작업 수를 계산합니다."""
        max_workers = cls.get_max_workers()
        sem_value = cls.get_gpu_semaphore_value()
        if isinstance(sem_value, int):
            return max_workers - sem_value
        return 0
    
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
        self._collection_cache_lock = threading.Lock()  # 컬렉션 캐시 동시성 제어
        
        # 텍스트 필드 최대 길이 설정
        self.MAX_TEXT_LENGTH = 9500  # 여유를 두고 9500으로 설정
    
    def get_collection(self, collection_name):
        """
        컬렉션을 효율적으로 관리합니다.
        - 이미 로드된 컬렉션은 캐시에서 반환
        - 접근 횟수를 추적하여 LRU(Least Recently Used) 방식으로 캐시 관리
        - 메모리 최적화를 위한 캐시 크기 제한
        - 스레드 안전성 보장
        """
        with self._collection_cache_lock:  # 캐시 접근 동기화
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
        """새 컬렉션을 로드하고 캐시에 추가합니다. (이미 락 내부에서 호출됨)"""
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
        대용량 데이터를 효율적으로 처리하기 위해 배치 단위로 삭제합니다.
        
        Args:
            domain (str): 삭제할 도메인
            doc_id (str): 삭제할 문서 ID
        """
        try:
            # 시간 로깅을 위한 로거 설정
            import logging
            timing_logger = logging.getLogger('timing')
            
            delete_start = time.time()
            print(f"[DEBUG] Deleting all passages for doc_id: {doc_id} in domain: {domain}")
            
            # doc_id가 이미 해시된 값인지 확인
            hash_start = time.time()
            is_already_hashed = len(doc_id) >= 64 and all(c in '0123456789abcdef' for c in doc_id.lower())
            hashed_doc_id = doc_id if is_already_hashed else self.data_p.hash_text(doc_id, hash_type='blake')
            hash_end = time.time()
            
            if not is_already_hashed:
                print(f"[DEBUG] Hashed doc_id from '{doc_id}' to '{hashed_doc_id}' in {hash_end - hash_start:.4f}s")
            
            # 컬렉션 로드
            load_start = time.time()
            collection = Collection(domain)
            collection.load()
            load_end = time.time()
            print(f"[DEBUG] Loaded collection in {load_end - load_start:.4f}s")
            
            # 삭제할 항목의 전체 개수 확인 (선택적)
            count_start = time.time()
            try:
                # 빠른 개수 체크
                count_result = collection.query(
                    expr=f'doc_id == "{hashed_doc_id}"',
                    output_fields=["count(*)"],
                    limit=1
                )
                total_items = count_result[0]["count(*)"] if count_result and "count(*)" in count_result[0] else None
            except Exception as count_error:
                # 개수 쿼리가 실패하면 대체 방법 시도
                try:
                    items = collection.query(
                        expr=f'doc_id == "{hashed_doc_id}"',
                        output_fields=["passage_id"],
                        limit=10000  # 안전한 상한값
                    )
                    total_items = len(items)
                except Exception as e:
                    print(f"[DEBUG] Count operation also failed: {str(e)}")
                    total_items = None
            count_end = time.time()
            
            if total_items is not None:
                print(f"[DEBUG] Found {total_items} items to delete (count in {count_end - count_start:.4f}s)")
                timing_logger.info(f"DELETE_COUNT - doc_id: {hashed_doc_id}, count: {total_items}, time: {count_end - count_start:.4f}s")
            
            # 효율적인 삭제 실행
            delete_expr = f'doc_id == "{hashed_doc_id}"'
            
            # DB 세마포어 획득
            self.__class__.init_db_semaphore()
            with self.__class__.db_semaphore:
                # 실제 삭제 수행
                exec_start = time.time()
                
                try:
                    # 단일 삭제 표현식으로 모든 항목 삭제
                    deleted_result = collection.delete(delete_expr)
                    
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
                    
                    # 즉시 변경사항 적용
                    collection.flush()
                    
                    exec_end = time.time()
                    exec_duration = exec_end - exec_start
                    delete_duration = exec_end - delete_start
                    
                    print(f"[DEBUG] Successfully deleted {deleted_count} items in {exec_duration:.4f}s (total: {delete_duration:.4f}s)")
                    timing_logger.info(f"DELETE_COMPLETE - doc_id: {hashed_doc_id}, count: {deleted_count}, time: {exec_duration:.4f}s, total: {delete_duration:.4f}s")
                    return True
                    
                except Exception as delete_error:
                    # 삭제 실패 시 대체 전략 시도
                    print(f"[DEBUG] Bulk delete failed: {str(delete_error)}. Trying alternative strategy...")
                    
                    try:
                        # 페이징을 통한 삭제 시도
                        alt_start = time.time()
                        batch_size = 1000
                        offset = 0
                        total_deleted = 0
                        max_iterations = 100  # 무한 루프 방지
                        
                        for i in range(max_iterations):
                            # 삭제할 항목의 ID 검색
                            items = collection.query(
                                expr=delete_expr,
                                output_fields=["passage_id"],
                                limit=batch_size,
                                offset=offset
                            )
                            
                            if not items:
                                break  # 더 이상 삭제할 항목 없음
                                
                            # 개별 항목 삭제
                            passage_ids = [f'passage_id == "{item["passage_id"]}"' for item in items if "passage_id" in item]
                            
                            if passage_ids:
                                batch_expr = " || ".join(passage_ids)
                                batch_result = collection.delete(batch_expr)
                                
                                # MutationResult 객체에서 삭제된 항목 수 추출
                                if hasattr(batch_result, 'delete_count'):
                                    batch_deleted = batch_result.delete_count
                                else:
                                    # 객체 자체가 정수일 경우 (이전 버전 호환) 또는 속성명이 다를 경우 대비
                                    try:
                                        batch_deleted = int(batch_result)
                                    except (TypeError, ValueError):
                                        # 다른 가능한 속성 이름 시도
                                        batch_deleted = getattr(batch_result, 'num_deleted', 0) or getattr(batch_result, 'count', 0)
                                
                                total_deleted += batch_deleted
                                
                                if (i + 1) % 5 == 0 or not items:  # 5번째 배치마다 flush
                                    collection.flush()
                                    
                                print(f"[DEBUG] Deleted batch {i+1}: {batch_deleted} items")
                            
                            # 남은 항목이 batch_size보다 적으면 종료
                            if len(items) < batch_size:
                                break
                        
                        # 마지막 flush
                        collection.flush()
                        
                        alt_end = time.time()
                        alt_duration = alt_end - alt_start
                        delete_duration = alt_end - delete_start
                        
                        print(f"[DEBUG] Successfully deleted {total_deleted} items with alternative method in {alt_duration:.4f}s (total: {delete_duration:.4f}s)")
                        timing_logger.info(f"DELETE_COMPLETE_ALT - doc_id: {hashed_doc_id}, count: {total_deleted}, time: {alt_duration:.4f}s, total: {delete_duration:.4f}s")
                        return True
                        
                    except Exception as alt_error:
                        # 대체 방법도 실패
                        exec_end = time.time()
                        exec_duration = exec_end - exec_start
                        delete_duration = exec_end - delete_start
                        
                        print(f"[ERROR] All delete methods failed: {str(alt_error)}")
                        timing_logger.error(f"DELETE_FAILED - doc_id: {hashed_doc_id}, error: {str(alt_error)}, time: {exec_duration:.4f}s, total: {delete_duration:.4f}s")
                        return False
            
        except Exception as e:
            print(f"[ERROR] Failed to delete passages for doc_id {doc_id}: {str(e)}")
            timing_logger.error(f"DELETE_ERROR - doc_id: {doc_id}, error: {str(e)}")
            return False

    # 전역 DB 접근 세마포어 및 배치 처리 관련 변수
    db_semaphore = None
    # 불필요한 변수 제거 - 글로벌 배치 큐로 대체됨
    batch_size = 300  # 기본 배치 크기 (300으로 통일)
    # 레거시 호환성을 위해 유지 (하지만 실제로는 사용하지 않음)
    batch_lock = None
    batch_data = {}
    
    # 중복 메서드 제거 (위에 이미 동일한 기능의 get_gpu_semaphore 메서드가 있음)
    
    @classmethod
    def init_batch_processing(cls):
        """배치 처리를 위한 공유 변수를 초기화합니다."""
        # 배치 크기 설정
        cls.batch_size = int(os.getenv('BATCH_SIZE', '300'))  # 100에서 300으로 증가
        print(f"[DEBUG] Initialized batch processing with size {cls.batch_size}")
        
        # 글로벌 배치 워커 시작
        cls.start_batch_worker()
        print(f"[DEBUG] Global batch worker started")
    
    @classmethod
    def init_db_semaphore(cls):
        """DB 접근을 제한하기 위한 세마포어를 초기화합니다."""
        if cls.db_semaphore is None:
            max_db_connections = int(os.getenv('MAX_DB_CONNECTIONS', '20'))  # 기본값: 20
            cls.db_semaphore = threading.BoundedSemaphore(max_db_connections)
            print(f"[DEBUG] Initialized DB semaphore with {max_db_connections} max connections")
    
    def insert_data(self, domain, doc_id, title, author, text, info, tags, ignore=True):
        try:
            # 시간 로깅을 위한 로거 설정
            import logging
            import threading
            timing_logger = logging.getLogger('timing')
            
            # DB 세마포어 및 배치 처리 락 초기화 (한 번만)
            if self.__class__.db_semaphore is None:
                max_db_connections = int(os.getenv('MAX_DB_CONNECTIONS', '20'))  # 최대 20개 동시 연결 기본값
                batch_size = int(os.getenv('BATCH_SIZE', '10'))  # 배치 크기 설정
                self.__class__.db_semaphore = threading.BoundedSemaphore(max_db_connections)
                self.__class__.batch_lock = threading.Lock()
                self.__class__.batch_size = batch_size
                print(f"[DEBUG] Initialized DB connection semaphore with max {max_db_connections} connections and batch size {batch_size}")
            
            print(f"[DEBUG] Original text length: {len(text)}")
            
            # doc_id 해시 처리
            hashed_doc_id = self.data_p.hash_text(doc_id, hash_type='blake')
            try:
                date = tags.get('date', '00000000').replace('-','')  # 날짜 없으면 기본값
                raw_doc_id = f"{date}-{title}-{author}"
                if len(raw_doc_id.encode('utf-8')) > 1024:
                    raw_doc_id = raw_doc_id[:200] + "..."  # 길이 제한
            except Exception as e:
                print(f"[WARNING] Error creating raw_doc_id: {str(e)}")
                raw_doc_id = f"unknown_doc_{hashed_doc_id[:8]}"  # 폴백
            print(f"[DEBUG] Hashed doc_id: {hashed_doc_id}, Raw doc_id: {raw_doc_id}")
            
            # 중복 문서 체크 - check_duplicates 함수 활용
            duplicate_results = self.check_duplicates([hashed_doc_id], domain)
            print(f"[DEBUG] 중복 체크 결과: {duplicate_results}")
            
            # 중복된 문서가 존재하는 경우
            if duplicate_results and hashed_doc_id in duplicate_results:
                existing_chunks = len(duplicate_results.get(hashed_doc_id, []))
                print(f"[DEBUG] Document with doc_id {hashed_doc_id} already exists in domain {domain} with at least {existing_chunks} chunks")
                timing_logger.info(f"DUPLICATE_FOUND - doc_id: {hashed_doc_id}, chunks: {existing_chunks}, ignore: {ignore}")
                
                if ignore:
                    print(f"[DEBUG] Skipping document due to ignore=True")
                    timing_logger.info(f"DUPLICATE_SKIPPED - doc_id: {hashed_doc_id}")
                    return "skipped"  # 중복으로 인한 건너뛰기 상태 반환
                else:
                    print(f"[DEBUG] Deleting existing document due to ignore=False")
                    
                    # 기존 문서 삭제 시작
                    delete_start_time = time.time()
                    timing_logger.info(f"DELETE_START - doc_id: {hashed_doc_id}, existing_chunks: {existing_chunks}")
                    
                    try:
                        # 기존 문서 삭제
                        delete_success = self.delete_data(domain, hashed_doc_id)
                        if not delete_success:
                            print(f"[ERROR] Failed to delete existing document")
                            return "error"
                        
                        delete_end_time = time.time()
                        delete_duration = delete_end_time - delete_start_time
                        timing_logger.info(f"DELETE_END - doc_id: {hashed_doc_id}, duration: {delete_duration:.4f}s")
                        print(f"[DEBUG] Successfully deleted existing document in {delete_duration:.4f}s")
                        
                    except Exception as delete_error:
                        delete_error_time = time.time()
                        delete_duration = delete_error_time - delete_start_time
                        timing_logger.error(f"DELETE_ERROR - doc_id: {hashed_doc_id}, duration: {delete_duration:.4f}s, error: {str(delete_error)}")
                        print(f"[ERROR] Failed to delete existing document: {str(delete_error)}")
                        return "error"
            
            # 텍스트 청킹 시작
            chunk_split_start = time.time()
            timing_logger.info(f"CHUNK_SPLIT_START - doc_id: {hashed_doc_id}")
            
            chunked_texts = self.data_p.chunk_text(text)
            
            chunk_split_end = time.time()
            chunk_split_duration = chunk_split_end - chunk_split_start
            timing_logger.info(f"CHUNK_SPLIT_END - doc_id: {hashed_doc_id}, chunks: {len(chunked_texts)}, duration: {chunk_split_duration:.4f}s")
            print(f"[DEBUG] Number of chunks: {len(chunked_texts)}")
            
            # info와 tags가 문자열인 경우 파싱
            if isinstance(info, str):
                info = json.loads(info)
            if isinstance(tags, str):
                tags = json.loads(tags)
            
            # 청크 처리를 위한 병렬 처리 함수
            def process_chunk(chunk_data):
                try:
                    i, (chunk, passage_id) = chunk_data
                    total_chunks = len(chunked_texts)
                    print(f"[DEBUG] Processing chunk {i+1}/{total_chunks} in thread")
                    
                    # 텍스트 길이 체크 - 새 알고리즘에서는 청크 크기가 최대 약 512바이트로 제한됨
                    chunk_bytes = len(chunk.encode('utf-8'))
                    if chunk_bytes > 512:  # 스키마 제약에 맞게 정확히 512바이트로 제한
                        error_msg = f"Text chunk too large: {chunk_bytes} bytes exceeds maximum 512 bytes"
                        print(f"[ERROR] {error_msg}")
                        raise ValueError(error_msg)
                    
                    # passage의 고유 식별자 생성
                    passage_uid = f"{hashed_doc_id}_{passage_id}"
                    
                    # 개별 청크 임베딩 시작
                    chunk_emb_start = time.time()
                    
                    # 임베딩 생성 (GPU 제한 적용됨)
                    chunk_emb = self.emb_model.bge_embed_data(chunk)
                    
                    chunk_emb_end = time.time()
                    chunk_emb_duration = chunk_emb_end - chunk_emb_start
                    
                    # 임베딩 결과 검증
                    if not chunk_emb or len(chunk_emb) == 0:
                        raise ValueError(f"Empty embedding generated for chunk {i+1}")
                    
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
                    # DB 삽입 시작 (배치 처리 사용)
                    db_insert_start = time.time()
                    print(f"[DEBUG] Preparing chunk {i+1} with passage_uid: {passage_uid} for batch insert")
                    
                    try:
                        # 배치에 추가하고 필요시 삽입
                        data_item = data[0]  # 단일 항목 추출
                        batch_inserted = self._add_to_batch_and_insert(data_item, domain)
                        
                        db_insert_end = time.time()
                        db_insert_duration = db_insert_end - db_insert_start
                        
                        if batch_inserted:
                            print(f"[DEBUG] Chunk {i+1} triggered a batch insert")
                        else:
                            print(f"[DEBUG] Chunk {i+1} added to batch (will be inserted later)")
                            
                        print(f"[TIMING] Chunk {i+1} - embedding: {chunk_emb_duration:.4f}s, batch_process: {db_insert_duration:.4f}s")
                        return f"chunk_{i+1}_success"
                        
                    except Exception as db_error:
                        db_insert_error_time = time.time()
                        db_insert_error_duration = db_insert_error_time - db_insert_start
                        error_msg = f"DB insert failed for chunk {i+1}: {str(db_error)} (duration: {db_insert_error_duration:.4f}s)"
                        print(f"[ERROR] {error_msg}")
                        raise Exception(error_msg)
                    
                except Exception as e:
                    error_msg = f"Error processing chunk {i+1}: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    raise Exception(error_msg)
            
            # 임베딩 및 DB 삽입 시작
            embedding_start_time = time.time()
            timing_logger.info(f"EMBEDDING_START - doc_id: {hashed_doc_id}, chunks: {len(chunked_texts)}")
            
            # 청크별 임베딩 생성을 병렬 처리 (환경변수로 설정, 기본 10개 스레드)
            max_workers = min(
                int(os.getenv('INSERT_CHUNK_THREADS', '10')),  # 기본값을 10으로 설정
                len(chunked_texts),  # 청크 수보다 많은 스레드는 불필요
            )
            print(f"[DEBUG] Using {max_workers} threads for chunk embedding processing (total chunks: {len(chunked_texts)})")
            
            chunk_start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 청크 데이터와 인덱스를 함께 전달
                chunk_data_list = [(i, chunk_data) for i, chunk_data in enumerate(chunked_texts)]
                
                # 모든 청크를 병렬로 처리
                future_to_chunk = {executor.submit(process_chunk, chunk_data): chunk_data for chunk_data in chunk_data_list}
                
                # 결과 수집 및 오류 처리
                successful_chunks = 0
                failed_chunks = 0
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_data = future_to_chunk[future]
                    chunk_index = chunk_data[0]
                    try:
                        result = future.result()
                        print(f"[DEBUG] {result}")
                        successful_chunks += 1
                    except Exception as exc:
                        failed_chunks += 1
                        error_msg = f"Chunk {chunk_index + 1} processing failed: {exc}"
                        print(f"[ERROR] {error_msg}")
                        # 하나라도 실패하면 전체 작업 중단
                        raise Exception(f"Insert failed - {error_msg}")
                
                print(f"[DEBUG] Chunk processing summary: {successful_chunks} successful, {failed_chunks} failed")
            
            chunk_end_time = time.time()
            embedding_duration = chunk_end_time - embedding_start_time
            timing_logger.info(f"EMBEDDING_END - doc_id: {hashed_doc_id}, duration: {embedding_duration:.4f}s")
            
            # 남은 배치 데이터 처리
            batch_flush_start = time.time()
            print(f"[DEBUG] Flushing any remaining batch data for domain: {domain}")
            # self._flush_batch(domain) 대신 글로벌 배치 큐에 추가
            remaining_batches = self._get_remaining_batches(domain)
            if remaining_batches:
                self.__class__.add_to_global_batch(remaining_batches, domain)
                print(f"[DEBUG] Added {len(remaining_batches)} remaining items to global batch queue for domain: {domain}")
            batch_flush_end = time.time()
            
            # DB 삽입 완료
            db_insert_end_time = time.time()
            db_insert_duration = db_insert_end_time - embedding_start_time
            timing_logger.info(f"DB_INSERT_END - doc_id: {hashed_doc_id}, duration: {db_insert_duration:.4f}s, batch_flush: {(batch_flush_end - batch_flush_start):.4f}s")
            
            print(f"[TIMING] 모든 청크 처리 완료: {(chunk_end_time - chunk_start_time):.4f}초, 배치 정리: {(batch_flush_end - batch_flush_start):.4f}초")
            
            return "success"  # 성공적인 삽입 상태 반환
                
        except ValueError as ve:
            timing_logger.error(f"INSERT_VALIDATION_ERROR - doc_id: {hashed_doc_id if 'hashed_doc_id' in locals() else 'unknown'}, error: {str(ve)}")
            print(f"[ERROR] Validation error: {str(ve)}")
            raise
        except Exception as e:
            error_time = time.time()
            if 'embedding_start_time' in locals():
                error_duration = error_time - embedding_start_time
                timing_logger.error(f"INSERT_TOTAL_ERROR - doc_id: {hashed_doc_id if 'hashed_doc_id' in locals() else 'unknown'}, duration: {error_duration:.4f}s, error: {str(e)}")
            else:
                timing_logger.error(f"INSERT_EARLY_ERROR - doc_id: {hashed_doc_id if 'hashed_doc_id' in locals() else 'unknown'}, error: {str(e)}")
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
            # 시간 로깅을 위한 로거 설정
            import logging
            timing_logger = logging.getLogger('timing')
            
            # raw_insert 전용 로거 설정
            raw_insert_logger = logging.getLogger('raw_insert')
            if not raw_insert_logger.handlers:
                # 로그 디렉토리 확인
                log_dir = "/var/log/rag" if os.path.exists("/var/log/rag") else "../logs"
                os.makedirs(log_dir, exist_ok=True)
                
                raw_handler = logging.FileHandler(os.path.join(log_dir, 'raw_insert.log'))
                raw_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                raw_handler.setFormatter(raw_formatter)
                raw_insert_logger.setLevel(logging.INFO)
                raw_insert_logger.addHandler(raw_handler)
                raw_insert_logger.propagate = False  # 다른 로거로 전파 방지
            
            raw_start_time = time.time()
            timing_logger.info(f"RAW_INSERT_START - doc_id: {doc_id}, passage_id: {passage_id}")
            raw_insert_logger.info(f"=== RAW_INSERT_START - doc_id: {doc_id}, passage_id: {passage_id} ===")
            
            # DB 세마포어 및 배치 처리 락 초기화 (한 번만)
            if self.__class__.db_semaphore is None:
                max_db_connections = int(os.getenv('MAX_DB_CONNECTIONS', '20'))  # 최대 20개 동시 연결 기본값
                batch_size = int(os.getenv('BATCH_SIZE', '10'))  # 배치 크기 설정
                self.__class__.db_semaphore = threading.BoundedSemaphore(max_db_connections)
                self.__class__.batch_lock = threading.Lock()
                self.__class__.batch_size = batch_size
                print(f"[DEBUG] Initialized DB connection semaphore with max {max_db_connections} connections and batch size {batch_size}")
            
            # 도메인이 없으면 생성
            if domain not in self.vectorenv.get_list_collection():
                print(f"[DEBUG] Creating new collection: {domain}")
                raw_insert_logger.info(f"Creating new collection: {domain}")
                self.create_domain(domain)
                print(f"[DEBUG] Collection created successfully")
            
            # doc_id는 사용자 입력값을 raw_doc_id로 사용하고 해시
            raw_doc_id = doc_id  # 사용자 입력값을 원본으로 저장
            if len(raw_doc_id.encode('utf-8')) > 1024:
                raise ValueError("doc_id is too long (max 1024 bytes)")
            
            hashed_doc_id = self.data_p.hash_text(doc_id, hash_type='blake')
            print(f"[DEBUG] Raw doc_id: {raw_doc_id}, Hashed doc_id: {hashed_doc_id}")
            raw_insert_logger.info(f"Raw doc_id: {raw_doc_id}, Hashed doc_id: {hashed_doc_id}")
            
            # 컬렉션 로드
            collection = Collection(domain)
            collection.load()
            print(f"[DEBUG] Collection loaded: {collection.name}, num_entities: {collection.num_entities}")
            
            # passage_uid 생성 (해시된 doc_id 사용)
            passage_uid = f"{hashed_doc_id}-p{passage_id}"
            raw_insert_logger.info(f"생성된 passage_uid: {passage_uid}")
            
            # 중복 체크 - check_duplicates 함수 활용
            duplicate_results = self.check_duplicates([hashed_doc_id], domain)
            raw_insert_logger.info(f"중복 체크 결과: 타입={type(duplicate_results)}, 값={duplicate_results}")
            print(f"[DEBUG] 중복 체크 결과: {duplicate_results}")
            
            # 수정된 중복 처리 로직 - 단순화 및 버그 수정
            is_update = False
            # duplicate_results는 리스트 형태로 반환됨
            if duplicate_results and hashed_doc_id in duplicate_results:
                raw_insert_logger.info(f"문서 ID {hashed_doc_id}가 중복 발견됨")
                
                # 중복 문서의 passage_id 직접 확인
                try:
                    passage_query = f'doc_id == "{hashed_doc_id}"'
                    passage_results = collection.query(
                        expr=passage_query,
                        output_fields=["passage_id", "passage_uid"],
                        limit=100  # 충분한 결과 확보
                    )
                    
                    raw_insert_logger.info(f"중복 문서의 passage 정보: {passage_results}")
                    
                    # 같은 passage_id가 있는지 확인
                    matching_passages = [p for p in passage_results if p.get('passage_id') == passage_id]
                    raw_insert_logger.info(f"일치하는 passage: {matching_passages}")
                    
                    if matching_passages:
                        if ignore:
                            print(f"[DEBUG] Skipping insert due to ignore=True")
                            raw_insert_logger.info(f"ignore=True로 인해 삽입 건너뜀")
                            timing_logger.info(f"RAW_INSERT_SKIPPED - doc_id: {hashed_doc_id}, passage_id: {passage_id}")
                            return "skipped"
                        else:
                            print(f"[DEBUG] Deleting existing document due to ignore=False")
                            raw_insert_logger.info(f"ignore=False로 인해 기존 문서 삭제 후 재삽입")
                            timing_logger.info(f"RAW_DELETE_START - doc_id: {hashed_doc_id}, passage_id: {passage_id}")
                            
                            # 삭제 작업 시작
                            delete_start = time.time()
                            try:
                                # passage_uid 기반 삭제가 doc_id & passage_id 쿼리보다 효율적
                                del_expr = f'passage_uid == "{passage_uid}"'
                                raw_insert_logger.info(f"삭제 쿼리: {del_expr}")
                                
                                deleted_result = collection.delete(del_expr)
                                
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
                                
                                raw_insert_logger.info(f"삭제 결과: {deleted_count}개 항목 삭제됨")
                                
                                delete_end = time.time()
                                delete_duration = delete_end - delete_start
                                timing_logger.info(f"RAW_DELETE_END - doc_id: {hashed_doc_id}, passage_id: {passage_id}, duration: {delete_duration:.4f}s")
                                print(f"[DEBUG] Successfully deleted existing document in {delete_duration:.4f}s")
                                
                                is_update = True
                            except Exception as delete_error:
                                delete_error_time = time.time()
                                delete_duration = delete_error_time - delete_start
                                timing_logger.error(f"RAW_DELETE_ERROR - doc_id: {hashed_doc_id}, passage_id: {passage_id}, duration: {delete_duration:.4f}s, error: {str(delete_error)}")
                                raw_insert_logger.error(f"삭제 실패: {str(delete_error)}")
                                print(f"[ERROR] Failed to delete existing document: {str(delete_error)}")
                                raise Exception(f"Raw delete operation failed: {str(delete_error)}")
                except Exception as query_error:
                    raw_insert_logger.error(f"passage 쿼리 오류: {str(query_error)}")
                    print(f"[ERROR] Error querying passages: {str(query_error)}")
            
            # 텍스트 임베딩
            embed_start = time.time()
            timing_logger.info(f"RAW_EMBED_START - doc_id: {hashed_doc_id}, passage_id: {passage_id}, text_length: {len(text)}")
            raw_insert_logger.info(f"임베딩 시작 - 텍스트 길이: {len(text)}")
            
            # GPU 세마포어를 사용하여 임베딩 생성
            with self.__class__.get_gpu_semaphore():
                text_emb = self.emb_model.bge_embed_data(text)
            
            embed_end = time.time()
            embed_duration = embed_end - embed_start
            timing_logger.info(f"RAW_EMBED_END - doc_id: {hashed_doc_id}, passage_id: {passage_id}, duration: {embed_duration:.4f}s")
            raw_insert_logger.info(f"임베딩 완료 - 소요시간: {embed_duration:.4f}초, 임베딩 벡터 길이: {len(text_emb)}")
            print(f"[DEBUG] Generated embedding, length: {len(text_emb)}")
            
            # info와 tags가 문자열인 경우 파싱
            if isinstance(info, str):
                info = json.loads(info)
            if isinstance(tags, str):
                tags = json.loads(tags)
            
            # 데이터 삽입 (해시된 doc_id 사용)
            data = {
                "passage_uid": passage_uid,
                "doc_id": hashed_doc_id,
                "raw_doc_id": raw_doc_id,
                "passage_id": passage_id,
                "domain": domain,
                "title": title,
                "author": author,
                "text": text,
                "text_emb": text_emb,
                "info": info,
                "tags": tags
            }
            
            # 배치 처리 메커니즘 사용
            insert_start = time.time()
            timing_logger.info(f"RAW_DB_INSERT_START - doc_id: {hashed_doc_id}, passage_id: {passage_id}")
            raw_insert_logger.info(f"DB 삽입 시작 - passage_uid: {passage_uid}")
            print(f"[DEBUG] Preparing data with passage_uid: {passage_uid} for insert")
            
            # 배치에 추가하고 필요시 삽입
            try:
                batch_inserted = self._add_to_batch_and_insert(data, domain)
                
                if batch_inserted:
                    print(f"[DEBUG] Data triggered a batch insert")
                    raw_insert_logger.info("배치 삽입 발생")
                else:
                    print(f"[DEBUG] Data added to batch (will be inserted later)")
                    raw_insert_logger.info("데이터가 배치에 추가됨 (나중에 삽입)")
                
                insert_end = time.time()
                insert_duration = insert_end - insert_start
                timing_logger.info(f"RAW_DB_INSERT_END - doc_id: {hashed_doc_id}, passage_id: {passage_id}, duration: {insert_duration:.4f}s")
                raw_insert_logger.info(f"DB 삽입 완료 - 소요시간: {insert_duration:.4f}초")
                
                # 업데이트 된 경우 즉시 flush하여 변경사항 적용
                if is_update:
                    flush_start = time.time()
                    raw_insert_logger.info("업데이트로 인한 flush 시작")
                    collection.flush()
                    flush_end = time.time()
                    timing_logger.info(f"RAW_FLUSH - doc_id: {hashed_doc_id}, passage_id: {passage_id}, duration: {(flush_end - flush_start):.4f}s")
                    raw_insert_logger.info(f"Flush 완료 - 소요시간: {(flush_end - flush_start):.4f}초")
                    print(f"[DEBUG] Collection flushed for update")
            except Exception as insert_error:
                insert_error_time = time.time()
                insert_duration = insert_error_time - insert_start
                timing_logger.error(f"RAW_DB_INSERT_ERROR - doc_id: {hashed_doc_id}, passage_id: {passage_id}, duration: {insert_duration:.4f}s, error: {str(insert_error)}")
                raw_insert_logger.error(f"DB 삽입 오류: {str(insert_error)}")
                print(f"[ERROR] Failed to insert data: {str(insert_error)}")
                raise
            
            # 인덱스가 있는지 확인하고 없으면 생성 (첫 삽입 시에만 필요)
            if not collection.has_index():
                print(f"[DEBUG] Creating index for collection")
                raw_insert_logger.info("컬렉션에 인덱스 생성")
                self.vectorenv.create_index(collection, field_name='text_emb')
                print(f"[DEBUG] Index created successfully")
                
                # 인덱스 생성 후 컬렉션 다시 로드
                collection.load()
                print(f"[DEBUG] Collection reloaded with index")
            
            raw_end_time = time.time()
            raw_duration = raw_end_time - raw_start_time
            timing_logger.info(f"RAW_INSERT_TOTAL - doc_id: {hashed_doc_id}, passage_id: {passage_id}, duration: {raw_duration:.4f}s, status: {'updated' if is_update else 'success'}")
            raw_insert_logger.info(f"=== RAW_INSERT_END - 총 소요시간: {raw_duration:.4f}초, 상태: {'updated' if is_update else 'success'} ===")
            print(f"[DEBUG] Raw insert completed in {raw_duration:.4f}s")
            
            return "updated" if is_update else "success"
            
        except Exception as e:
            print(f"Error in raw_insert_data: {str(e)}")
            # 예외 발생 시에도 로그 기록
            import logging
            raw_insert_logger = logging.getLogger('raw_insert')
            raw_insert_logger.error(f"심각한 오류: {str(e)}")
            import traceback
            raw_insert_logger.error(f"스택 트레이스: {traceback.format_exc()}")
            raise
    
    # 배치 삽입 처리 메소드 (새로 추가)
    def _add_to_batch_and_insert(self, data, domain):
        """
        데이터를 배치에 추가하고 배치 크기가 충족되면 삽입합니다.
        배치 처리를 통해 DB 접근 횟수를 줄입니다.
        
        글로벌 배치 큐를 사용하도록 수정됨
        
        Args:
            data (dict 또는 list): 삽입할 데이터 항목 (딕셔너리) 또는 항목 리스트
            domain (str): 삽입할 도메인(컬렉션)
        
        Returns:
            bool: 배치 삽입이 수행되었는지 여부
        """
        # 글로벌 배치 큐 사용
        batch_full = self.__class__.add_to_global_batch(data, domain)
        
        # 하위 호환성을 위해 배치 크기 충족 여부 반환
        return batch_full
    
    def _execute_batch_insert(self, batch_data, domain):
        """배치 데이터 삽입을 위한 개선된 메서드"""
        try:
            import gc
            import logging
            logger = logging.getLogger('rag-backend')
            
            # 로그 추가: 삽입 데이터 확인
            logger.info(f"배치 삽입 시작: 총 {len(batch_data)}개 항목, 도메인: {domain}")
            
            collection = self.get_collection(domain)
            max_batch_size = 50  # 한 번에 처리할 최대 레코드 수
            total_success = 0
            
            insert_start = time.time()
            
            # 유효한 데이터만 필터링 (중요한 필드와 text_emb가 있는지 확인)
            valid_batch_data = []
            for item in batch_data:
                # 필수 필드 확인
                required_fields = ['passage_uid', 'doc_id', 'text']
                if not all(field in item for field in required_fields):
                    logger.warning(f"필수 필드 누락 항목 건너뜀: {item.get('passage_uid', 'unknown')}")
                    continue
                
                # 임베딩 벡터 확인 (반드시 필요)
                if 'text_emb' not in item or not item['text_emb']:
                    logger.warning(f"임베딩 벡터 없음, 항목 건너뜀: {item.get('passage_uid', 'unknown')}")
                    continue
                
                # 임베딩 벡터 간단 검증 (models.py에서 이미 표준 리스트로 변환되어 있다고 가정)
                # 벡터 타입만 확인하고 리스트가 아니면 오류 로그만 남김
                if not isinstance(item['text_emb'], list):
                    logger.error(f"임베딩 벡터가 리스트 타입이 아님: {type(item['text_emb'])}, passage_uid={item.get('passage_uid', 'unknown')}")
                    continue
                
                # 벡터 길이 확인 (1024가 아니면 오류 로그만 남김)
                if len(item['text_emb']) != 1024:
                    logger.error(f"임베딩 벡터 길이가 1024가 아님: {len(item['text_emb'])}, passage_uid={item.get('passage_uid', 'unknown')}")
                    continue
                
                valid_batch_data.append(item)
            
            # 유효한 데이터가 없으면 종료
            if not valid_batch_data:
                logger.warning(f"유효한 데이터 항목이 없습니다. 삽입 취소. 원본 데이터 개수: {len(batch_data)}")
                return False
            
            logger.info(f"유효한 배치 데이터: {len(valid_batch_data)}/{len(batch_data)}개")
            
            for i in range(0, len(valid_batch_data), max_batch_size):
                sub_batch = valid_batch_data[i:i + max_batch_size]
                
                # 각 레코드의 텍스트 길이 검증
                final_batch = []
                for item in sub_batch:
                    # 텍스트 길이 검증
                    text_length = len(item.get('text', '').encode('utf-8'))
                    if text_length > self.MAX_TEXT_LENGTH:
                        # 텍스트가 너무 긴 경우 분할
                        sub_chunks = self._split_large_chunk(item['text'])
                        for idx, sub_chunk in enumerate(sub_chunks):
                            new_item = item.copy()
                            new_item['text'] = sub_chunk
                            # passage_id가 문자열인 경우 정수로 변환 시도
                            try:
                                base_passage_id = int(item['passage_id']) if not isinstance(item['passage_id'], int) else item['passage_id']
                                new_item['passage_id'] = base_passage_id * 100 + idx  # 하위 청크에 대해 고유한 passage_id 생성
                            except (ValueError, TypeError):
                                # 변환 실패 시 인덱스 사용
                                new_item['passage_id'] = idx
                            # passage_uid 업데이트
                            new_item['passage_uid'] = f"{item['passage_uid']}_{idx}"
                            
                            # 최종 검증 (text_emb는 이미 검증 완료됨)
                            final_batch.append(new_item)
                    else:
                        # passage_id가 문자열이면 정수로 변환
                        if 'passage_id' in item and not isinstance(item['passage_id'], int):
                            try:
                                item['passage_id'] = int(item['passage_id'])
                            except (ValueError, TypeError):
                                # 변환 실패 시 기본값 사용
                                item['passage_id'] = 0
                        
                        final_batch.append(item)
                
                try:
                    # 배치 삽입 시도
                    if final_batch:
                        # 로그에 첫 항목의 벡터 타입 정보 추가 (디버깅용)
                        if final_batch and 'text_emb' in final_batch[0]:
                            first_emb = final_batch[0]['text_emb']
                            logger.info(f"첫 항목 text_emb 정보: 타입={type(first_emb)}, 길이={len(first_emb) if hasattr(first_emb, '__len__') else 'unknown'}")
                        
                        # 실제 삽입 시도
                        collection.insert(final_batch)
                        total_success += len(final_batch)
                        logger.info(f"배치 {i//max_batch_size + 1} 삽입 성공: {len(final_batch)}개 항목")
                        
                        # 매 배치마다 flush하지 않고, 최종 단계에서만 flush
                        # 즉시 flush 코드 제거
                    
                    # 주기적으로 메모리 정리만 하고 flush는 최종에만 수행
                    if (i + max_batch_size) % (max_batch_size * 4) == 0:
                        gc.collect()
                        
                except Exception as batch_error:
                    logger.warning(f"배치 삽입 실패, 개별 삽입 시도: {str(batch_error)}")
                    
                    # 개별 삽입 시도 - 간소화된 버전
                    for item in final_batch:
                        try:
                            # 단일 항목 검증 (임베딩 벡터만 간단히 확인)
                            if not isinstance(item.get('text_emb', []), list):
                                logger.error(f"개별 항목 text_emb가 리스트가 아님: {type(item.get('text_emb', []))}, 항목 건너뜀")
                                continue
                                
                            # 단일 항목 삽입
                            collection.insert([item])
                            total_success += 1
                            logger.info(f"개별 항목 삽입 성공: {item.get('passage_uid', 'unknown')}")
                        except Exception as item_error:
                            logger.error(f"개별 항목 삽입 실패: {str(item_error)}, 항목: {item.get('passage_uid', 'unknown')}")
                            
                    # 개별 삽입 후 flush
                    try:
                        collection.flush()
                    except Exception as flush_error:
                        logger.warning(f"개별 삽입 후 flush 실패: {str(flush_error)}")
                    
            # 최종 flush 진행
            try:
                collection.flush()
                logger.info(f"최종 flush 완료")
            except Exception as flush_error:
                logger.warning(f"최종 flush 실패: {str(flush_error)}")
            
            # 추가 검증: flush 후 데이터가 저장되었는지 확인
            try:
                # 이 배치에서 삽입한 임의의 passage_uid 선택
                if valid_batch_data and len(valid_batch_data) > 0:
                    sample_passage_uid = valid_batch_data[0].get('passage_uid')
                    if sample_passage_uid:
                        # 실제로 저장되었는지 확인
                        verify_expr = f'passage_uid == "{sample_passage_uid}"'
                        verify_results = collection.query(expr=verify_expr, output_fields=["passage_uid"], limit=1)
                        if verify_results and len(verify_results) > 0:
                            logger.info(f"데이터 저장 확인 성공: passage_uid={sample_passage_uid}")
                        else:
                            logger.warning(f"데이터 저장 확인 실패: passage_uid={sample_passage_uid}가 데이터베이스에 없음")
                            # 다시 flush 시도
                            logger.info("데이터베이스 다시 flush 시도")
                            collection.flush()
            except Exception as verify_error:
                logger.error(f"데이터 저장 확인 중 오류: {str(verify_error)}")
            
            insert_end = time.time()
            insert_duration = insert_end - insert_start
            
            logger.info(f"배치 삽입 완료: {total_success}/{len(batch_data)}개 성공, 소요시간: {insert_duration:.4f}초")
            
            return total_success > 0
            
        except Exception as e:
            logger.error(f"배치 삽입 중 예외 발생: {str(e)}")
            import traceback
            logger.error(f"스택 트레이스: {traceback.format_exc()}")
            return False
    
    def flush_all_batches(self):
        """모든 도메인의 배치 데이터를 즉시 삽입합니다."""
        with self.__class__.batch_lock:
            for domain, batch_data in self.__class__.batch_data.items():
                if batch_data:
                    # 배치 데이터 복사 및 비우기
                    data_to_insert = batch_data.copy()
                    self.__class__.batch_data[domain] = []
                    
                    # 락 해제 상태에서 삽입 수행
                    self._execute_batch_insert(data_to_insert, domain)

    def _get_remaining_batches(self, domain):
        """
        특정 도메인의 배치 큐에 남아있는 데이터를 가져옵니다.
        글로벌 배치 큐 사용을 위한 호환성 메서드입니다.
        
        Args:
            domain (str): 데이터를 가져올 도메인
            
        Returns:
            list: 배치 데이터 목록 또는 빈 리스트
        """
        # 더 이상 인스턴스 배치 데이터를 사용하지 않으므로 빈 리스트 반환
        return []
    
    def embed_and_prepare_chunk(self, chunk_data):
        """청크 데이터에 임베딩을 추가하고 DB 삽입을 위한 데이터로 변환합니다."""
        try:
            import logging
            self.insert_logger = logging.getLogger('insert')
            
            # 시작 시간 기록
            start_time = time.time()
            
            # 스레드 ID 및 청크 메타데이터 추출
            thread_id = threading.get_ident()
            chunk_index = chunk_data.get('chunk_index', 0)
            chunk_no = chunk_data.get('chunk_no', None)
            
            # 청크 텍스트 추출
            chunk_text = chunk_data.get('text', '')
            
            # 텍스트 타입 확인 및 변환 (튜플 지원)
            if isinstance(chunk_text, tuple):
                if len(chunk_text) > 0:
                    chunk_text = chunk_text[0]  # 튜플의 첫 번째 요소만 사용
                else:
                    self.insert_logger.warning(f"[Thread-{thread_id}] 빈 튜플 건너뛰기: 청크#{chunk_no if chunk_no is not None else chunk_index}")
                    return None
                self.insert_logger.info(f"[Thread-{thread_id}] 튜플 형태의 청크 텍스트 변환: 청크#{chunk_no if chunk_no is not None else chunk_index}")
            
            # 텍스트가 비어있으면 처리하지 않음
            if not chunk_text or len(chunk_text.strip()) == 0:
                self.insert_logger.warning(f"[Thread-{thread_id}] 빈 청크 건너뛰기: 청크#{chunk_no if chunk_no is not None else chunk_index}")
                return None
            
            # 세마포어 획득 시작 시간
            sem_wait_start = time.time()
            
            # GPU 세마포어 획득 - 클래스 메서드 사용
            with InteractManager.get_gpu_semaphore():
                # 세마포어 획득 대기 시간 측정
                sem_wait_time = time.time() - sem_wait_start
                
                # GPU 자원 상태 로깅
                max_workers = InteractManager.get_max_workers()
                sem_value = InteractManager.get_gpu_semaphore_value()
                active_workers = InteractManager.get_active_workers()
                
                self.insert_logger.info(f"[Thread-{thread_id}] GPU 자원 상태: 활성작업={active_workers}/{max_workers}, 세마포어값={sem_value}, 대기시간={sem_wait_time:.4f}초 (청크#{chunk_no if chunk_no is not None else chunk_index})")
                
                # 임베딩 시작 시간
                embed_start = time.time()
                
                # 임베딩 생성
                embeddings = self.emb_model.bge_embed_data(chunk_text)
                
                # 임베딩 소요 시간
                embed_time = time.time() - embed_start
            
            # 결과 데이터 준비 - 원본 데이터의 ID 정보 유지
            prepared_data = {
                'id': chunk_data.get('id', f'chunk_{chunk_index}'),  # 원본 ID 유지 또는 생성
                'doc_id': chunk_data.get('doc_id', ''),
                'embedding': embeddings,
                'text': chunk_text,
                'metadata': chunk_data.get('metadata', {}),
                'chunk_index': chunk_index,  # 청크 인덱스 정보 전달
                'chunk_no': chunk_no  # 청크 번호 정보 전달
            }
            
            # 성능 로깅
            end_time = time.time()
            total_time = end_time - start_time
            
            # 비정상적으로 긴 처리 시간 경고 (10초 이상)
            if total_time > 10:
                self.insert_logger.warning(f"[Thread-{thread_id}] ⚠️ 비정상적으로 긴 청크 처리 시간: {total_time:.2f}초 (청크#{chunk_no if chunk_no is not None else chunk_index})")
            
            # 임베딩 완료 로그
            self.insert_logger.info(f"[Thread-{thread_id}] 청크 임베딩 완료: 총 {total_time:.4f}초, 대기={sem_wait_time:.4f}초, 계산={embed_time:.4f}초 (청크#{chunk_no if chunk_no is not None else chunk_index})")
            
            return prepared_data
            
        except Exception as e:
            # 오류 로깅
            error_trace = traceback.format_exc()
            self.insert_logger.error(f"[Thread-{thread_id}] 청크 임베딩 오류: {str(e)} (청크#{chunk_no if 'chunk_no' in locals() and chunk_no is not None else chunk_index})\n{error_trace}")
            return None

    def prepare_data_with_embedding(self, chunk_data):
        """
        app.py에서 호출하는 청크 임베딩 생성 함수
        내부적으로 embed_and_prepare_chunk를 호출하여 임베딩 작업 수행
        """
        try:
            import logging
            logger = logging.getLogger('rag-backend')
            insert_logger = logging.getLogger('insert')
            
            thread_id = threading.get_ident()
            
            # 필수 필드 확인 (간소화)
            if not all(field in chunk_data for field in ['text', 'doc_id', 'passage_uid']):
                insert_logger.error(f"[Thread-{thread_id}] 필수 필드 누락: {[f for f in ['text', 'doc_id', 'passage_uid'] if f not in chunk_data]}")
                return None
            
            # 텍스트 간소화된 전처리
            chunk_text = chunk_data.get('text', '')
            if isinstance(chunk_text, tuple):
                if len(chunk_text) > 0:
                    chunk_data['text'] = chunk_text[0]  # 튜플의 첫 번째 요소만 사용
                else:
                    insert_logger.error(f"[Thread-{thread_id}] 빈 튜플 텍스트: {chunk_data.get('passage_uid', 'unknown')}")
                    return None
            
            if not isinstance(chunk_data['text'], str):
                try:
                    chunk_data['text'] = str(chunk_data['text'])
                except Exception as e:
                    insert_logger.error(f"[Thread-{thread_id}] 텍스트 변환 실패: {str(e)}")
                    return None
            
            # 텍스트가 비어 있는지 확인
            if not chunk_data['text'] or len(chunk_data['text'].strip()) == 0:
                insert_logger.error(f"[Thread-{thread_id}] 빈 텍스트: {chunk_data.get('passage_uid', 'unknown')}")
                return None
            
            # passage_id 확인 및 정수형 변환 (간소화)
            if 'passage_id' in chunk_data and not isinstance(chunk_data['passage_id'], int):
                try:
                    chunk_data['passage_id'] = int(chunk_data['passage_id'])
                except (ValueError, TypeError):
                    # 변환 실패 시 메타데이터에서 chunk_index를 가져오거나 기본값 사용
                    if 'metadata' in chunk_data and 'chunk_index' in chunk_data['metadata']:
                        chunk_data['passage_id'] = int(chunk_data['metadata']['chunk_index'])
                    else:
                        chunk_data['passage_id'] = 0
            
            # 임베딩 생성 - bge_embed_data 함수에서 GPU 작업 추적 처리
            try:
                # 로그 정보 추가 (최소화)
                chunk_id = chunk_data.get('passage_uid', chunk_data.get('id', 'unknown'))
                
                # 임베딩 생성 (실제 임베딩은 bge_embed_data 함수에서 처리)
                emb_vector = self.emb_model.bge_embed_data(chunk_data['text'])
                
                # 임베딩 결과 저장
                chunk_data['text_emb'] = emb_vector
                
                # 임베딩 완료 로그 (최소화)
                emb_sample = str(emb_vector[:2])[:20] + "..." if len(emb_vector) > 0 else "비어있음"
                insert_logger.info(f"[Thread-{thread_id}] 최종 데이터 검증: passage_id={chunk_data.get('passage_id', 'unknown')}, 타입={type(chunk_data.get('passage_id', 0))}, 벡터 타입={type(chunk_data['text_emb'])}")
                
                return chunk_data
                
            except Exception as emb_error:
                insert_logger.error(f"[Thread-{thread_id}] 임베딩 생성 실패: {str(emb_error)}")
                return None
                
        except Exception as e:
            import logging, traceback
            insert_logger = logging.getLogger('insert')
            error_trace = traceback.format_exc()
            insert_logger.error(f"[Thread-{thread_id}] 청크 임베딩 처리 오류: {str(e)}\n{error_trace}")
            return None

    def batch_insert_data(self, domain, data_batch):
        """
        준비된 데이터 배치를 지정된 도메인에 삽입합니다.
        
        Args:
            domain (str): 삽입할 도메인 이름
            data_batch (list): 삽입할 데이터 항목의 리스트
            
        Returns:
            bool: 삽입 성공 여부
        """
        try:
            import logging
            logger = logging.getLogger('rag-backend')
            
            if not data_batch:
                logger.warning(f"빈 데이터 배치가 전달되었습니다 (도메인: {domain})")
                return False
            
            # None 값 필터링 (임베딩 실패한 항목)
            valid_batch = [item for item in data_batch if item is not None]
            
            # 원본 배치와 유효 배치의 길이 비교 로깅
            if len(valid_batch) < len(data_batch):
                logger.warning(f"None 항목 필터링: {len(valid_batch)}/{len(data_batch)}개 유효 항목 (도메인: {domain})")
            
            if len(valid_batch) == 0:
                logger.warning(f"유효한 데이터 항목이 없습니다 (도메인: {domain})")
                return False
            
            # 필수 필드 정의
            required_fields = ['passage_uid', 'doc_id', 'raw_doc_id', 'passage_id', 'domain', 'text', 'text_emb']
            
            # 필수 필드 확인 및 추가 - 모든 항목에 대해 확인
            valid_items = []
            for item in valid_batch:
                missing_fields = []
                
                # raw_doc_id 필드 확인 및 추가
                if 'raw_doc_id' not in item and 'doc_id' in item:
                    item['raw_doc_id'] = item['doc_id']
                    logger.info(f"항목에 raw_doc_id 필드 자동 추가: {item.get('passage_uid', 'unknown')}")
                
                # passage_id 필드 확인 및 추가
                if 'passage_id' not in item:
                    # passage_uid에서 ID 부분 추출 시도
                    passage_uid = item.get('passage_uid', '')
                    if '_' in passage_uid:
                        # passage_uid가 "doc_id_chunk_index" 형식인 경우, 마지막 부분 추출
                        try:
                            item['passage_id'] = int(passage_uid.split('_')[-1])
                            logger.info(f"항목에 passage_id 필드 자동 추가 (passage_uid에서 추출): {passage_uid} -> {item['passage_id']}")
                        except (ValueError, IndexError):
                            # 추출 실패 시 chunk_index 또는 기본값 사용
                            item['passage_id'] = int(item.get('chunk_index', 0))
                            logger.info(f"항목에 passage_id 필드 자동 추가 (기본값): {item['passage_id']}")
                    else:
                        # passage_uid에서 추출할 수 없는 경우 chunk_index 또는 기본값 사용
                        item['passage_id'] = int(item.get('chunk_index', 0))
                        logger.info(f"항목에 passage_id 필드 자동 추가 (기본값): {item['passage_id']}")
                else:
                    # passage_id가 있지만 정수형이 아닌 경우 변환
                    if not isinstance(item['passage_id'], int):
                        try:
                            item['passage_id'] = int(item['passage_id'])
                            logger.info(f"항목의 passage_id를 정수형으로 변환: {item['passage_id']}")
                        except (ValueError, TypeError):
                            # 변환 실패 시 기본값으로 설정
                            item['passage_id'] = int(item.get('chunk_index', 0))
                            logger.warning(f"항목의 passage_id를 정수형으로 변환 실패, 기본값 사용: {item['passage_id']}")
                
                # domain 필드가 없으면 현재 도메인 사용
                if 'domain' not in item:
                    item['domain'] = domain
                    logger.info(f"항목에 domain 필드 자동 추가: {domain}")
                
                # 필수 필드 확인
                missing_fields = [field for field in required_fields if field not in item]
                
                # 임베딩 벡터 유효성 검증 (필드는 있지만 빈 경우)
                if 'text_emb' in item and (item['text_emb'] is None or len(item['text_emb']) == 0):
                    missing_fields.append('text_emb (비어있음)')
                    logger.warning(f"항목 {item.get('passage_uid', 'unknown')}의 text_emb 필드가 비어 있습니다")
                
                # 모든 필수 필드가 있는 경우만 유효 항목으로 포함
                if not missing_fields:
                    valid_items.append(item)
                else:
                    logger.warning(f"필수 필드 누락으로 항목 제외: {item.get('passage_uid', 'unknown')}, 누락 필드: {missing_fields}")
            
            # 필수 필드 검증 후 유효 항목 개수 확인
            if len(valid_items) == 0:
                logger.warning(f"필수 필드 검증 후 유효한 항목이 없습니다 (도메인: {domain})")
                return False
            
            # 필터링 결과 로깅
            if len(valid_items) < len(valid_batch):
                logger.warning(f"필수 필드 검증 결과: {len(valid_items)}/{len(valid_batch)}개 항목만 유효함 (도메인: {domain})")
            
            # 상세 로깅 - 첫 번째 아이템의 키 확인
            if valid_items:
                sample_item = valid_items[0]
                # 중요 필드 확인 로그
                has_uid = 'passage_uid' in sample_item
                has_doc_id = 'doc_id' in sample_item
                has_raw_doc_id = 'raw_doc_id' in sample_item
                has_passage_id = 'passage_id' in sample_item
                has_text = 'text' in sample_item
                has_text_emb = 'text_emb' in sample_item
                has_domain = 'domain' in sample_item
                
                # 중요 필드 로깅
                logger.info(f"배치 삽입 검증 - 필수 필드 존재 여부: passage_uid={has_uid}, doc_id={has_doc_id}, raw_doc_id={has_raw_doc_id}, passage_id={has_passage_id}, domain={has_domain}, text={has_text}, text_emb={has_text_emb}")
                
                # 임베딩 벡터 확인
                if has_text_emb:
                    emb_length = len(sample_item['text_emb']) if sample_item['text_emb'] else 0
                    logger.info(f"임베딩 벡터 샘플 길이: {emb_length}")
            
            # 배치 삽입 수행
            logger.info(f"배치 삽입 시작: {len(valid_items)}개 유효 항목 (도메인: {domain})")
            insert_result = self._execute_batch_insert(valid_items, domain)
            
            # 삽입 결과 로깅
            if insert_result:
                logger.info(f"배치 삽입 성공 (도메인: {domain})")
            else:
                logger.warning(f"배치 삽입 실패 (도메인: {domain})")
            
            return insert_result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"배치 삽입 실패 (도메인: {domain}): {str(e)}\n{error_trace}")
            return False

    def chunk_document(self, text):
        """문서를 청크로 나누는 메서드"""
        try:
            # 입력이 딕셔너리일 경우 text 필드 추출
            if isinstance(text, dict) and 'text' in text:
                print(f"[DEBUG] 입력이 딕셔너리입니다. text 필드를 추출합니다.")
                text = text['text']
            
            # 입력이 문자열인지 확인
            if not isinstance(text, str):
                print(f"[ERROR] 문서 청킹 실패: 입력이 문자열이 아닙니다. 타입: {type(text)}")
                raise TypeError(f"입력은 문자열이어야 합니다. 받은 타입: {type(text)}")
            
            return self.data_p.chunk_text(text)
        except Exception as e:
            print(f"[ERROR] 문서 청킹 실패: {str(e)}")
            raise

    def _split_large_chunk(self, chunk, max_length=None):
        """큰 청크를 최대 길이에 맞게 분할"""
        if max_length is None:
            max_length = self.MAX_TEXT_LENGTH
            
        # 청크가 최대 길이보다 작으면 그대로 반환
        if len(chunk.encode('utf-8')) <= max_length:
            return [chunk]
            
        # 문장 단위로 분할
        sentences = chunk.split('.')
        current_chunk = ""
        chunks = []
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            temp_chunk = current_chunk + ' ' + sentence if current_chunk else sentence
            
            if len(temp_chunk.encode('utf-8')) > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = temp_chunk
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def check_duplicates(self, doc_ids, domain):
        """중복 문서 체크 - 리스트 형태로 중복된 doc_id만 반환"""
        try:
            if not doc_ids:
                print(f"[DUPLICATION_CHECK] 경고: 검사할 문서가 없습니다 (도메인: {domain})")
                return []
                
            # 중복 검사 전용 로깅 설정
            import logging
            duplication_logger = logging.getLogger('duplication')
            if not duplication_logger.handlers:
                # 로그 디렉토리 확인 - 상위 클래스 또는 전역 변수 활용
                log_dir = "/var/log/rag" if os.path.exists("/var/log/rag") else "../logs"
                os.makedirs(log_dir, exist_ok=True)
                
                duplication_handler = logging.FileHandler(os.path.join(log_dir, 'duplication.log'))
                duplication_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                duplication_handler.setFormatter(duplication_formatter)
                duplication_logger.setLevel(logging.INFO)
                duplication_logger.addHandler(duplication_handler)
                duplication_logger.propagate = False  # 다른 로거로 전파 방지
            
            duplication_logger.info(f"중복 검사 시작: 총 {len(doc_ids)}개 문서 ID, 도메인: {domain}")
            duplication_logger.debug(f"검사할 doc_ids: {doc_ids[:5]}{'...' if len(doc_ids) > 5 else ''}")
            
            print(f"[DUPLICATION_CHECK] 시작: 총 {len(doc_ids)}개 문서 ID 중복 검사 (도메인: {domain})")
            
            # 전체 doc_id 목록 로깅 (50개 미만일 경우)
            if len(doc_ids) <= 50:
                print(f"[DUPLICATION_CHECK] 검사할 모든 doc_id 목록: {doc_ids}")
            else:
                sample_ids = doc_ids[:5] + ['...'] + doc_ids[-5:]
                print(f"[DUPLICATION_CHECK] 검사할 doc_id 샘플: {sample_ids}")
                
            start_time = time.time()
            duplicates = []  # 중복된 문서 ID 저장 리스트
            errors = []      # 오류 정보 저장 리스트
            
            # 컬렉션 얻기
            try:
                collection = self.get_collection(domain)
                print(f"[DUPLICATION_CHECK] 컬렉션 '{domain}' 로드 완료")
                duplication_logger.info(f"컬렉션 '{domain}' 로드 완료")

                # 디버깅: 기존 문서의 doc_id 샘플 확인
                try:
                    sample_docs = collection.query(
                        expr="",  # 모든 문서
                        output_fields=["doc_id"],
                        limit=10,
                        offset=0
                    )
                    if sample_docs:
                        doc_id_samples = [doc.get('doc_id', 'unknown') for doc in sample_docs]
                        print(f"[DUPLICATION_CHECK] 기존 문서 doc_id 샘플 (10개): {doc_id_samples}")
                        duplication_logger.info(f"기존 문서 doc_id 샘플 (10개): {doc_id_samples}")
                    else:
                        # 컬렉션이 비어있음
                        print(f"[DUPLICATION_CHECK] 컬렉션 '{domain}'이 비어 있습니다. 중복 문서가 없습니다.")
                        duplication_logger.info(f"컬렉션 '{domain}'이 비어 있음, 중복 없음")
                        return []
                except Exception as e:
                    print(f"[DUPLICATION_CHECK] 기존 문서 샘플 확인 실패: {str(e)}")
                    duplication_logger.error(f"기존 문서 샘플 확인 실패: {str(e)}")
                
                # 실제 중복 체크 로직 추가
                batch_size = 100
                
                # 효율성을 위해 배치 단위로 처리
                for i in range(0, len(doc_ids), batch_size):
                    batch = doc_ids[i:i + batch_size]
                    if not batch:
                        continue
                        
                    # IN 연산자를 사용하여 배치로 쿼리
                    # 각 ID를 큰따옴표로 묶고 쉼표로 구분
                    ids_str = ", ".join([f'"{id}"' for id in batch])
                    expr = f'doc_id in [{ids_str}]'  # 올바른 IN 연산자 형식 사용: doc_id in ["id1", "id2", ...]
                    
                    # 쿼리 표현식을 INFO 레벨로 변경하여 항상 로그에 표시되도록 함
                    duplication_logger.info(f"배치 {i//batch_size + 1} 쿼리 실행: {expr[:100]}{'...' if len(expr) > 100 else ''}")
                    print(f"[DUPLICATION_CHECK] 배치 {i//batch_size + 1} 쿼리 실행 중")
                    
                    try:
                        # 존재하는 doc_id 가져오기
                        results = collection.query(
                            expr=expr,
                            output_fields=["doc_id", "passage_id"],
                            limit=10000  # 충분히 큰 값으로 설정
                        )
                        
                        # 쿼리 결과 정보 추가 로깅
                        duplication_logger.info(f"배치 {i//batch_size + 1} 쿼리 결과: {len(results)}개 항목 반환됨")
                        print(f"[DUPLICATION_CHECK] 배치 {i//batch_size + 1} 쿼리 결과: {len(results)}개 항목")
                        
                        # 결과 샘플 로깅 (최대 3개)
                        if results:
                            sample_results = results[:3]
                            duplication_logger.info(f"결과 샘플: {sample_results}")
                            print(f"[DUPLICATION_CHECK] 결과 샘플: {sample_results}")
                        
                        # 결과에서 중복 doc_id 추출
                        found_doc_ids = set()
                        for result in results:
                            doc_id = result.get('doc_id')
                            if doc_id and doc_id not in found_doc_ids:
                                found_doc_ids.add(doc_id)
                                if doc_id not in duplicates:
                                    duplicates.append(doc_id)
                                    
                        duplication_logger.info(f"배치 {i//batch_size + 1} 처리 완료: {len(found_doc_ids)}개 중복 발견")
                        print(f"[DUPLICATION_CHECK] 배치 {i//batch_size + 1} 처리 완료: {len(found_doc_ids)}개 중복 발견")
                        
                        # 중복 발견 항목 상세 로깅
                        if found_doc_ids:
                            duplication_logger.debug(f"배치 {i//batch_size + 1} 중복 ID: {list(found_doc_ids)[:10]}{'...' if len(found_doc_ids) > 10 else ''}")
                    except Exception as e:
                        print(f"[DUPLICATION_CHECK] 배치 {i//batch_size + 1} 처리 중 오류: {str(e)}")
                        duplication_logger.error(f"배치 {i//batch_size + 1} 처리 중 오류: {str(e)}")
                
                # 개별 체크 (배치 처리에서 누락된 경우를 대비)
                for doc_id in doc_ids:
                    # 이미 중복으로 확인된 ID는 건너뛰기
                    if doc_id in duplicates:
                        continue
                    
                    try:
                        # 개별 doc_id 쿼리
                        expr = f'doc_id == "{doc_id}"'
                        results = collection.query(
                            expr=expr,
                            output_fields=["passage_id"],
                            limit=1  # 존재 여부만 확인하면 됨
                        )
                        
                        if results:
                            # 문서가 존재하면 중복으로 표시
                            duplicates.append(doc_id)
                            print(f"[DUPLICATION_CHECK] 개별 확인: doc_id={doc_id}는 중복됨")
                            duplication_logger.info(f"개별 확인: doc_id={doc_id}는 중복됨")
                    except Exception as e:
                        # 오류 수집하여 나중에 분석
                        errors.append({"doc_id": doc_id, "error": str(e)})
                        print(f"[DUPLICATION_CHECK] 오류: doc_id={doc_id} 검사 실패: {str(e)}")
                        duplication_logger.error(f"오류: doc_id={doc_id} 검사 실패: {str(e)}")
            except Exception as e:
                print(f"[DUPLICATION_CHECK] 컬렉션 로드 오류: {str(e)}")
                duplication_logger.error(f"컬렉션 로드 오류: {str(e)}")
                return []
            
            total_time = time.time() - start_time
            print(f"[DUPLICATION_CHECK] 완료: 총 {len(doc_ids)}개 문서 중 {len(duplicates)}개 중복 발견, {len(errors)}개 검사 실패")
            print(f"[DUPLICATION_CHECK] 성능: 총 소요시간 {total_time:.2f}초, 문서당 평균 {total_time/len(doc_ids):.4f}초")
            duplication_logger.info(f"완료: 총 {len(doc_ids)}개 문서 중 {len(duplicates)}개 중복 발견, {len(errors)}개 검사 실패")
            duplication_logger.info(f"성능: 총 소요시간 {total_time:.2f}초, 문서당 평균 {total_time/len(doc_ids):.4f}초")
            
            # 에러가 있는 경우 상세 로깅
            if errors:
                print(f"[DUPLICATION_CHECK] 오류 상세: {errors[:5]}" + ("..." if len(errors) > 5 else ""))
                duplication_logger.error(f"오류 상세: {errors[:5]}" + ("..." if len(errors) > 5 else ""))
            
            # 중복 문서 목록 출력 (최대 50개)
            if duplicates:
                if len(duplicates) <= 50:
                    print(f"[DUPLICATION_CHECK] 중복 문서 ID 전체 목록: {duplicates}")
                    duplication_logger.info(f"중복 문서 ID 전체 목록: {duplicates}")
                else:
                    display_dupes = duplicates[:20]
                    more_count = len(duplicates) - len(display_dupes)
                    print(f"[DUPLICATION_CHECK] 중복 문서 ID 일부: {display_dupes} 외 {more_count}개")
                    duplication_logger.info(f"중복 문서 ID 일부: {display_dupes} 외 {more_count}개")
            else:
                print(f"[DUPLICATION_CHECK] 중복 문서 없음")
                duplication_logger.info(f"중복 문서 없음")
            
            # 중요: 반환 값 유형 명확하게 로깅
            print(f"[DUPLICATION_CHECK] 반환 값 유형: {type(duplicates)}, 값: {duplicates[:5] if len(duplicates) > 5 else duplicates}")
            duplication_logger.info(f"반환 값 유형: {type(duplicates)}, 값: {duplicates[:5] if len(duplicates) > 5 else duplicates}")
            
            return duplicates
            
        except Exception as e:
            print(f"[DUPLICATION_CHECK] 심각한 오류: 중복 체크 실패: {str(e)}")
            import traceback
            print(f"[DUPLICATION_CHECK] 스택 트레이스: {traceback.format_exc()}")
            
            # 중복 검사 로그에도 기록
            import logging
            duplication_logger = logging.getLogger('duplication')
            duplication_logger.error(f"심각한 오류: 중복 체크 실패: {str(e)}")
            duplication_logger.error(f"스택 트레이스: {traceback.format_exc()}")
            
            return []

    @classmethod
    def start_batch_worker(cls):
        """배치 처리 워커 스레드를 시작합니다."""
        if cls.batch_worker_thread is None or not cls.batch_worker_running:
            cls.batch_worker_running = True
            cls.batch_worker_thread = threading.Thread(target=cls._batch_worker_loop, daemon=True)
            cls.batch_worker_thread.start()
            print(f"[DEBUG] 배치 처리 워커 스레드 시작됨")

    @classmethod
    def stop_batch_worker(cls):
        """배치 처리 워커 스레드를 중지합니다."""
        cls.batch_worker_running = False
        if cls.batch_worker_thread and cls.batch_worker_thread.is_alive():
            cls.batch_worker_thread.join(timeout=2.0)
            print(f"[DEBUG] 배치 처리 워커 스레드 중지됨")

    @classmethod
    def _batch_worker_loop(cls):
        """배치 처리 워커 스레드의 메인 루프"""
        import logging
        logger = logging.getLogger('rag-backend')
        logger.info("배치 처리 워커 스레드 시작")
        
        while cls.batch_worker_running:
            try:
                # 각 도메인에 대해 배치 처리 확인
                domains_to_process = []
                
                with cls.global_batch_lock:
                    # 처리할 도메인 목록 생성
                    for domain, chunks in cls.global_batch_queue.items():
                        # 배치가 꽉 찬 경우에만 처리
                        if len(chunks) >= cls.batch_size:
                            domains_to_process.append(domain)
                
                # 처리할 도메인이 있으면 배치 처리 수행
                for domain in domains_to_process:
                    with cls.global_batch_lock:
                        # 배치 데이터 가져오기
                        if domain in cls.global_batch_queue and cls.global_batch_queue[domain]:
                            # 배치 크기만큼 또는 모든 청크 가져오기
                            if len(cls.global_batch_queue[domain]) <= cls.batch_size:
                                batch_data = cls.global_batch_queue[domain]
                                cls.global_batch_queue[domain] = []
                            else:
                                batch_data = cls.global_batch_queue[domain][:cls.batch_size]
                                cls.global_batch_queue[domain] = cls.global_batch_queue[domain][cls.batch_size:]
                            
                            # 타임스탬프 갱신 (로깅용으로만 사용)
                            cls.last_batch_time[domain] = time.time()
                            
                            batch_size = len(batch_data)
                            logger.info(f"글로벌 배치 처리: 도메인={domain}, 청크 수={batch_size}")
                        else:
                            continue
                    
                    # 락 해제 상태에서 배치 처리 (다른 스레드가 큐에 추가할 수 있도록)
                    if batch_data:
                        try:
                            # 인스턴스 생성 필요 (클래스 메서드에서 인스턴스 메서드 호출)
                            instance = InteractManager()
                            instance._execute_batch_insert(batch_data, domain)
                            logger.info(f"글로벌 배치 {batch_size}개 항목 처리 완료 (도메인: {domain})")
                        except Exception as e:
                            logger.error(f"글로벌 배치 처리 오류: {str(e)}")
                            # 오류 발생 시 개별 처리 시도
                            try:
                                for item in batch_data:
                                    try:
                                        instance._execute_batch_insert([item], domain)
                                    except Exception as item_error:
                                        logger.error(f"개별 항목 처리 실패: {str(item_error)}")
                            except Exception as recovery_error:
                                logger.error(f"복구 시도 중 오류: {str(recovery_error)}")
            
                # 처리할 도메인이 없으면 잠시 대기
                if not domains_to_process:
                    time.sleep(0.1)  # CPU 사용률 감소를 위한 짧은 대기
                    
            except Exception as e:
                logger.error(f"배치 워커 루프 오류: {str(e)}")
                time.sleep(1.0)  # 오류 발생 시 더 긴 대기
        
        logger.info("배치 처리 워커 스레드 종료")

    @classmethod
    def add_to_global_batch(cls, data, domain):
        """
        글로벌 배치 큐에 데이터를 추가합니다.
        
        Args:
            data (dict 또는 list): 추가할 데이터 항목 또는 항목 리스트
            domain (str): 도메인
            
        Returns:
            bool: 성공 여부
        """
        try:
            # 배치 워커 시작 확인
            if not cls.batch_worker_running:
                cls.start_batch_worker()
            
            # 단일 항목을 리스트로 변환
            if isinstance(data, dict):
                data_items = [data]
            else:
                data_items = data
            
            if not data_items:
                return False
            
            with cls.global_batch_lock:
                # 도메인별 큐 초기화
                if domain not in cls.global_batch_queue:
                    cls.global_batch_queue[domain] = []
                
                # 데이터 항목 추가
                cls.global_batch_queue[domain].extend(data_items)
                
                # 마지막 배치 추가 시간 갱신 (로깅용)
                cls.last_batch_time[domain] = time.time()
                
                # 배치 크기 이상이 되면 배치 처리가 필요함을 표시
                return len(cls.global_batch_queue[domain]) >= cls.batch_size
                
        except Exception as e:
            import logging
            logger = logging.getLogger('rag-backend')
            logger.error(f"글로벌 배치 큐 추가 실패: {str(e)}")
            return False

    @classmethod
    def flush_all_batches(cls):
        """글로벌 배치 큐의 모든 데이터를 처리합니다."""
        import logging
        logger = logging.getLogger('rag-backend')
        logger.info("모든 배치 데이터 강제 처리 시작")
        
        try:
            domains_to_process = []
            batch_data_map = {}
            
            # 모든 배치 데이터를 안전하게 가져오기
            with cls.global_batch_lock:
                for domain, chunks in cls.global_batch_queue.items():
                    if chunks:
                        domains_to_process.append(domain)
                        batch_data_map[domain] = chunks.copy()
                        cls.global_batch_queue[domain] = []
            
            if not domains_to_process:
                logger.info("처리할 배치 데이터가 없음")
                return
            
            # 모든 도메인의 배치 데이터 처리
            for domain in domains_to_process:
                batch_data = batch_data_map[domain]
                if batch_data:
                    logger.info(f"도메인 {domain}의 {len(batch_data)}개 항목 처리 중")
                    try:
                        # 인스턴스 생성 필요 (클래스 메서드에서 인스턴스 메서드 호출)
                        instance = InteractManager()
                        instance._execute_batch_insert(batch_data, domain)
                        logger.info(f"도메인 {domain} 배치 데이터 처리 완료")
                    except Exception as e:
                        logger.error(f"도메인 {domain} 배치 데이터 처리 실패: {str(e)}")
                        # 중요 데이터이므로 항목별 처리 시도
                        success_count = 0
                        for item in batch_data:
                            try:
                                instance._execute_batch_insert([item], domain)
                                success_count += 1
                            except Exception as item_error:
                                logger.error(f"항목 처리 실패: {str(item_error)}")
                        logger.info(f"도메인 {domain} 항목별 처리 결과: {success_count}/{len(batch_data)}개 성공")
            
            logger.info("모든 배치 데이터 처리 완료")
        except Exception as e:
            logger.error(f"배치 데이터 강제 처리 중 오류 발생: {str(e)}")
            import traceback
            logger.error(f"오류 상세: {traceback.format_exc()}")