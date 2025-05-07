"""
Milvus 컬렉션 초기 설정 스크립트 (Collection Initialization Script)
==================================================================

이 파일은 Milvus 데이터베이스의 초기 컬렉션 구조를 설정하는 스크립트입니다.
pipe.py의 InteractManager.create_domain 함수와 함께 시스템의 핵심 스키마를 정의합니다.

주의사항:
- 이 파일은 삭제하면 안 됩니다! 시스템 초기화에 필요한 코어 컴포넌트입니다.
- pipe.py와의 필드 정의가 일치해야 합니다. 특히 필드 타입과 길이 제한이 중요합니다.
- 필드 길이를 변경할 경우 양쪽 파일(set_collection.py와 pipe.py)에 모두 적용해야 합니다.

시스템 구조:
- set_collection.py: 초기 컬렉션 생성 및 기본 인덱스 설정 (CLI 실행용)
- pipe.py의 create_domain: 런타임 중 새 도메인(컬렉션) 생성 (API 요청으로 실행)

컬렉션 생성 흐름:
1. 시스템 초기화 시 set_collection.py로 기본 컬렉션 생성
2. 새 도메인 필요 시 InteractManager.create_domain 호출하여 동적 생성
"""

from src import milvus, MilvusEnvManager
from dotenv import load_dotenv
import argparse
import json
import os 


def main(args):
    """메인 함수: Milvus 컬렉션 초기화 및 생성

    Parameters:
        args: 명령줄 인자 (config_path, config_name, collection_name)
    """
    load_dotenv()
    ip_addr = os.getenv('ip_addr')

    with open(os.path.join(args.config_path, args.config_name)) as f:
        db_args = json.load(f)
    db_args['ip_addr'] = ip_addr

    milvus_db = MilvusEnvManager(db_args)
    print(f'ip: {milvus_db.ip_addr}')

    milvus_db.set_env()
    print(f'client: {milvus_db.client}')
    
    # 필드 정의 (중요: pipe.py의 create_domain 함수와 일치해야 함)
    # 필드 길이를 수정하는 경우 양쪽 파일 모두 업데이트 필요
    data_doc_id = milvus_db.create_field_schema('doc_id', dtype='VARCHAR', is_primary=True, max_length=1024)
    data_passage_id = milvus_db.create_field_schema('passage_id', dtype='INT64')
    data_domain = milvus_db.create_field_schema('domain', dtype='VARCHAR', max_length=32)
    data_title = milvus_db.create_field_schema('title', dtype='VARCHAR', max_length=1024)  # 제목 길이: 최대 1024바이트
    data_author = milvus_db.create_field_schema('author', dtype='VARCHAR', max_length=128)
    data_text = milvus_db.create_field_schema('text', dtype='VARCHAR', max_length=512)   # 500B (500글자 단위로 문서 분할)
    data_text_emb = milvus_db.create_field_schema('text_emb', dtype='FLOAT_VECTOR', dim=1024)
    data_info = milvus_db.create_field_schema('info', dtype='JSON')
    data_tags = milvus_db.create_field_schema('tags', dtype='JSON')
    schema_field_list = [data_doc_id, data_passage_id, data_domain, data_title, data_author, data_text, data_text_emb, data_info, data_tags]

    # 스키마 및 컬렉션 생성
    schema = milvus_db.create_schema(schema_field_list, 'schema for fai-rag, using fastcgi')
    collection = milvus_db.create_collection(args.collection_name, schema, shards_num=2)
    milvus_db.get_collection_info(args.collection_name)
    milvus_db.create_index(collection, field_name='text_emb')   # text_emb 필드에 인덱스 생성

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='./config/', help='설정 파일 경로')
    cli_parser.add_argument('--config_name', type=str, default='db_config.json', help='설정 파일 이름')
    cli_parser.add_argument('--collection_name', type=str, default=None, help='생성할 컬렉션 이름')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)
