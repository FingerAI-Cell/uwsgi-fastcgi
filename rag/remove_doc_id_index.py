#!/usr/bin/env python3
"""
모든 Milvus 컬렉션에서 doc_id 인덱스를 제거하는 스크립트
"""

from pymilvus import connections, utility, Collection, MilvusException
import json
import os
import time
import logging
import sys

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
        # 먼저 상위 디렉토리에 config 폴더가 있는지 확인
        config_path = 'config/db_config.json'
        parent_config_path = '../config/db_config.json'
        
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"설정 파일 로드 성공: {config_path}")
        elif os.path.exists(parent_config_path):
            with open(parent_config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"설정 파일 로드 성공: {parent_config_path}")
        
        # 환경 변수에서 IP 주소 가져오기 (db_config.json에는 ip_addr이 없음)
        ip_addr = os.environ.get('ip_addr', 'milvus-server')  # localhost 대신 milvus-server를 기본값으로 사용
        logger.info(f"Milvus 서버 IP: {ip_addr} (환경 변수에서 로드)")
        
        # config에 ip_addr 추가
        config['ip_addr'] = ip_addr
        
        return config
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {str(e)}")
        raise

def connect_to_milvus(config):
    """Milvus 연결"""
    try:
        # 먼저 이전 연결이 있으면 정리
        try:
            connections.disconnect("default")
            logger.info("기존 연결 종료")
        except:
            pass
        
        # 연결 정보 로그
        host = config.get('ip_addr', 'milvus-server')
        port = config.get('port', '19530')
        logger.info(f"Milvus 서버 연결 시도: {host}:{port}")
        
        # 연결 시도 - 타임아웃 설정 추가
        connections.connect(
            alias="default", 
            host=host,
            port=port,
            timeout=5.0  # 5초 타임아웃 설정
        )
        
        # 연결 확인
        if utility.has_collection("_test_connectivity_"):
            logger.info("연결 확인: 성공 (컬렉션 확인 가능)")
        else:
            all_collections = utility.list_collections()
            logger.info(f"연결 확인: 성공 (서버에 연결됨, 컬렉션 수: {len(all_collections)})")
            if all_collections:
                logger.info(f"사용 가능한 컬렉션: {', '.join(all_collections)}")
        
        return True
    except MilvusException as me:
        error_code = getattr(me, 'code', '알 수 없음')
        error_msg = str(me)
        logger.error(f"Milvus 연결 실패 (Milvus 오류): 코드={error_code}, 메시지={error_msg}")
        
        # 연결 문제 가능성 진단
        if "connection" in error_msg.lower():
            logger.error(f"연결 문제 가능성: 1) Milvus 서버가 실행 중이 아닙니다. 2) 호스트명({host})이 잘못되었습니다. 3) 네트워크 연결 문제가 있습니다.")
            logger.error(f"도커 환경의 경우: 컨테이너 간 통신은 'localhost' 대신 컨테이너 이름이나 네트워크 IP를 사용해야 합니다.")
        
        return False
    except Exception as e:
        logger.error(f"Milvus 연결 실패 (일반 오류): {str(e)}")
        return False

def remove_doc_id_index():
    """모든 컬렉션에서 doc_id 인덱스 제거"""
    try:
        # 모든 컬렉션 조회
        collections = utility.list_collections()
        if not collections:
            logger.warning("컬렉션이 없습니다.")
            return True
            
        logger.info(f"총 {len(collections)}개 컬렉션 발견: {', '.join(collections)}")
        
        success_count = 0
        fail_count = 0
        skip_count = 0

        for coll_name in collections:
            try:
                logger.info(f"컬렉션 처리 중: {coll_name}")
                
                # 컬렉션 로드
                collection = Collection(coll_name)
                collection.load()
                
                # 모든 인덱스 조회
                try:
                    indexes = collection.index().info
                    logger.info(f"컬렉션 '{coll_name}'에 {len(indexes) if indexes else 0}개 인덱스 존재")
                except Exception as idx_error:
                    logger.error(f"인덱스 정보 조회 실패: {str(idx_error)}")
                    indexes = []
                
                # doc_id 관련 인덱스 찾기
                doc_id_indexes = []
                
                for idx in indexes:
                    if idx.get('field_name') == 'doc_id':
                        doc_id_indexes.append(idx)
                        logger.info(f"doc_id 인덱스 발견: {idx.get('index_name', 'unnamed')}")
                
                # doc_id 인덱스가 없으면 건너뛰기
                if not doc_id_indexes:
                    logger.info(f"컬렉션 '{coll_name}'에 doc_id 인덱스가 없습니다. 건너뜁니다.")
                    skip_count += 1
                    continue
                    
                # 인덱스 삭제
                index_success = True
                for idx in doc_id_indexes:
                    index_name = idx.get('index_name')
                    if index_name:
                        logger.info(f"인덱스 '{index_name}' 삭제 시도...")
                        try:
                            collection.drop_index(index_name=index_name)
                            logger.info(f"인덱스 '{index_name}' 삭제 완료")
                        except Exception as e:
                            logger.error(f"인덱스 '{index_name}' 삭제 실패: {str(e)}")
                            index_success = False
                
                # 컬렉션 변경사항 적용
                try:
                    collection.flush()
                    logger.info(f"컬렉션 '{coll_name}' 변경사항 적용 완료")
                except Exception as flush_error:
                    logger.error(f"컬렉션 '{coll_name}' 변경사항 적용 실패: {str(flush_error)}")
                    index_success = False
                    
                # 결과 집계
                if index_success:
                    success_count += 1
                    logger.info(f"컬렉션 '{coll_name}' 처리 완료")
                else:
                    fail_count += 1
                    logger.warning(f"컬렉션 '{coll_name}' 처리 중 일부 오류 발생")
                
            except Exception as coll_error:
                logger.error(f"컬렉션 '{coll_name}' 처리 중 오류: {str(coll_error)}")
                fail_count += 1
                continue
        
        logger.info(f"모든 컬렉션 처리 완료 (성공: {success_count}, 실패: {fail_count}, 건너뜀: {skip_count})")
        return fail_count == 0
    
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
            sys.exit(1)
        
        # 인덱스 제거
        if remove_doc_id_index():
            logger.info("인덱스 제거 작업 성공")
        else:
            logger.error("인덱스 제거 작업 실패")
            sys.exit(2)
    
    except Exception as e:
        logger.error(f"예상치 못한 오류: {str(e)}")
        sys.exit(3)
    
    finally:
        # 연결 종료
        try:
            connections.disconnect("default")
            logger.info("Milvus 연결 종료")
        except:
            pass
        
        end_time = time.time()
        logger.info(f"작업 완료. 총 소요시간: {(end_time - start_time):.2f}초")

if __name__ == "__main__":
    main() 