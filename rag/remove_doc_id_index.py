#!/usr/bin/env python3
"""
모든 Milvus 컬렉션에서 doc_id 인덱스를 제거하는 스크립트
"""

from pymilvus import connections, utility, Collection
import json
import os
import time
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('remove_doc_id_index.log')
    ]
)
logger = logging.getLogger('remove-index')

def load_config():
    """DB 설정 로드"""
    try:
        with open('config/db_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {str(e)}")
        raise

def connect_to_milvus(config):
    """Milvus 연결"""
    try:
        connections.connect(
            alias="default", 
            host=config.get('ip_addr', 'localhost'),
            port=config.get('port', '19530')
        )
        logger.info(f"Milvus 서버 연결 성공: {config.get('ip_addr', 'localhost')}:{config.get('port', '19530')}")
        return True
    except Exception as e:
        logger.error(f"Milvus 연결 실패: {str(e)}")
        return False

def remove_doc_id_index():
    """모든 컬렉션에서 doc_id 인덱스 제거"""
    try:
        # 모든 컬렉션 조회
        collections = utility.list_collections()
        logger.info(f"총 {len(collections)}개 컬렉션 발견")

        for coll_name in collections:
            try:
                logger.info(f"컬렉션 처리 중: {coll_name}")
                
                # 컬렉션 로드
                collection = Collection(coll_name)
                
                # 모든 인덱스 조회
                indexes = collection.index().info
                logger.info(f"컬렉션 '{coll_name}'에 {len(indexes) if indexes else 0}개 인덱스 존재")
                
                # doc_id 관련 인덱스 찾기
                doc_id_indexes = []
                
                for idx in indexes:
                    if idx.get('field_name') == 'doc_id':
                        doc_id_indexes.append(idx)
                        logger.info(f"doc_id 인덱스 발견: {idx.get('index_name', 'unnamed')}")
                
                # 인덱스 삭제
                for idx in doc_id_indexes:
                    index_name = idx.get('index_name')
                    if index_name:
                        logger.info(f"인덱스 '{index_name}' 삭제 시도...")
                        try:
                            collection.drop_index(index_name=index_name)
                            logger.info(f"인덱스 '{index_name}' 삭제 완료")
                        except Exception as e:
                            logger.error(f"인덱스 '{index_name}' 삭제 실패: {str(e)}")
                
                # 컬렉션 변경사항 적용
                collection.flush()
                logger.info(f"컬렉션 '{coll_name}' 처리 완료")
                
            except Exception as coll_error:
                logger.error(f"컬렉션 '{coll_name}' 처리 중 오류: {str(coll_error)}")
                continue
        
        logger.info("모든 컬렉션 처리 완료")
        return True
    
    except Exception as e:
        logger.error(f"인덱스 제거 중 오류 발생: {str(e)}")
        return False

def main():
    """메인 함수"""
    logger.info("doc_id 인덱스 제거 작업 시작")
    start_time = time.time()
    
    try:
        # 설정 로드
        config = load_config()
        
        # Milvus 연결
        if not connect_to_milvus(config):
            logger.error("Milvus 연결 실패로 작업 중단")
            return
        
        # 인덱스 제거
        if remove_doc_id_index():
            logger.info("인덱스 제거 작업 성공")
        else:
            logger.error("인덱스 제거 작업 실패")
    
    except Exception as e:
        logger.error(f"예상치 못한 오류: {str(e)}")
    
    finally:
        # 연결 종료
        try:
            connections.disconnect("default")
            logger.info("Milvus 연결 종료")
        except:
            pass
        
        end_time = time.time()
        logger.info(f"작업 완료. 총 소요시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    main() 