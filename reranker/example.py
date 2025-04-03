"""
RAG 시스템과 Reranker 통합 예제
"""

import os
import sys
import json
from typing import Dict, List, Any

# 현재 디렉토리를 패스에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# RAG 모듈 임포트
from rag.src.pipe import InteractManager
from rag.src.data_p import DataProcessor
from rag.src.milvus import MilvusEnvManager, DataMilVus, MilvusMeta
from rag.src.models import EmbModel

# Reranker 모듈 임포트
from reranker.service import RerankerService


def setup_rag():
    """RAG 시스템 설정"""
    
    # 기본 설정
    args = {
        'config_path': 'rag/config',
        'db_config': 'db_config.json',
        'llm_config': 'llm_config.json',
        'ip_addr': 'localhost'
    }
    
    # 환경 설정
    env = InteractManager()
    
    # RAG 시스템 로드
    try:
        # 환경 관리자 로드 (실제 구현 시 필요한 파라미터 추가)
        data_p = DataProcessor()
        vectorenv = MilvusEnvManager({"ip_addr": args['ip_addr']})
        vectordb = DataMilVus({"ip_addr": args['ip_addr']})
        
        # 임베딩 모델 설정
        emb_model = EmbModel({})
        emb_model.set_emb_model(model_type='bge')
        emb_model.set_embbeding_config()
        
        # InteractManager 인스턴스 생성
        rag_manager = InteractManager(
            data_p=data_p,
            vectorenv=vectorenv,
            vectordb=vectordb,
            emb_model=emb_model
        )
        
        print("[INFO] RAG 시스템이 성공적으로 로드되었습니다.")
        return rag_manager
        
    except Exception as e:
        print(f"[ERROR] RAG 시스템 로드 중 오류 발생: {str(e)}")
        return None


def search_and_rerank(query: str, top_k: int = 5):
    """
    쿼리로 검색한 후 결과를 재순위화
    
    Args:
        query: 검색 쿼리
        top_k: 반환할 결과 수
    """
    # RAG 시스템 설정
    rag_manager = setup_rag()
    if not rag_manager:
        print("[ERROR] RAG 시스템을 설정할 수 없습니다.")
        return
    
    # Reranker 설정
    reranker = RerankerService()
    
    try:
        # 검색 수행
        print(f"[INFO] 쿼리 실행 중: '{query}'")
        search_results = rag_manager.retrieve_data(
            query=query,
            top_k=top_k * 3,  # 재순위화를 위해 더 많은 결과 검색
            filter_conditions=None
        )
        
        if not search_results:
            print("[INFO] 검색 결과가 없습니다.")
            return []
            
        print(f"[INFO] {len(search_results)}개의 결과를 검색했습니다.")
        
        # 재순위화 수행
        reranked_results = reranker.rerank_passages(query, search_results, top_k=top_k)
        
        print(f"[INFO] 재순위화 완료. 상위 {len(reranked_results)}개 결과:")
        
        # 결과 출력
        for i, result in enumerate(reranked_results):
            print(f"\n--- 결과 #{i+1} (점수: {result.get('rerank_score', 0):.4f}) ---")
            print(f"제목: {result.get('title', '제목 없음')}")
            print(f"텍스트: {result.get('text', '').strip()[:200]}...")
            
        return reranked_results
        
    except Exception as e:
        print(f"[ERROR] 검색 및 재순위화 중 오류 발생: {str(e)}")
        return []


def get_document_with_reranking(doc_id: str):
    """
    문서 ID로 문서를 검색하고 패시지를 재순위화
    
    Args:
        doc_id: 문서 ID
    """
    # RAG 시스템 설정
    rag_manager = setup_rag()
    if not rag_manager:
        print("[ERROR] RAG 시스템을 설정할 수 없습니다.")
        return
    
    # Reranker 설정
    reranker = RerankerService()
    
    try:
        # 문서 검색
        print(f"[INFO] 문서 ID 검색 중: '{doc_id}'")
        doc_info = rag_manager.get_document_passages(doc_id)
        
        if not doc_info:
            print(f"[INFO] 문서 ID '{doc_id}'에 대한 정보를 찾을 수 없습니다.")
            return None
            
        # 재순위화를 위한 쿼리 생성 (문서 제목 사용)
        query = doc_info.get("title", "")
        
        if not query:
            print("[INFO] 재순위화를 위한 쿼리를 생성할 수 없습니다.")
            return doc_info
            
        # 패시지 추출 및 재순위화
        passages = doc_info.get("passages", [])
        
        if not passages:
            print("[INFO] 재순위화할 패시지가 없습니다.")
            return doc_info
            
        print(f"[INFO] {len(passages)}개 패시지 재순위화 중...")
        
        # 재순위화 수행
        reranked_passages = reranker.rerank_passages(query, passages)
        
        # 원본 문서에 재순위화된 패시지 업데이트
        doc_info["passages"] = reranked_passages
        doc_info["reranked"] = True
        
        print(f"[INFO] 패시지 재순위화 완료")
        
        return doc_info
        
    except Exception as e:
        print(f"[ERROR] 문서 검색 및 재순위화 중 오류 발생: {str(e)}")
        return None


if __name__ == "__main__":
    # 예제 1: 검색 및 재순위화
    print("\n=== 예제 1: 검색 및 재순위화 ===")
    search_and_rerank("메타버스 기술의 발전", top_k=3)
    
    # 예제 2: 문서 패시지 재순위화
    print("\n=== 예제 2: 문서 패시지 재순위화 ===")
    doc_id = "20220804-메타버스 뉴스"
    doc_info = get_document_with_reranking(doc_id)
    
    if doc_info:
        print(f"\n문서: {doc_info['title']}")
        print("재순위화된 패시지:")
        for i, passage in enumerate(doc_info['passages'][:3]):
            print(f"\n{i+1}. (점수: {passage.get('rerank_score', 0):.4f})")
            print(f"텍스트: {passage['text'][:200]}...") 