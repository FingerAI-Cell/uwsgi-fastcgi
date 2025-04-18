"""
RAG 시스템과 Reranker 통합 예제
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
import requests

# 현재 디렉토리를 패스에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# RAG 관련 함수
def search_rag(query: str, top_k: int = 10) -> dict:
    """
    RAG 시스템을 사용하여 검색 수행
    
    Args:
        query: 검색 쿼리
        top_k: 검색 결과 수
        
    Returns:
        검색 결과
    """
    try:
        url = f"http://localhost/search?query_text={query}&top_k={top_k}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] RAG 검색 중 오류: {e}")
        return {"results": []}


def get_document(doc_id: str) -> dict:
    """
    문서 ID로 문서 검색
    
    Args:
        doc_id: 문서 ID
        
    Returns:
        문서 정보
    """
    try:
        url = f"http://localhost/document?doc_id={doc_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] 문서 검색 중 오류: {e}")
        return {}


# Reranker 관련 함수
def rerank_results(query: str, passages: List[Dict[str, Any]], top_k: Optional[int] = None) -> dict:
    """
    검색 결과 재순위화
    
    Args:
        query: 검색 쿼리
        passages: 검색된 패시지 목록
        top_k: 반환할 결과 수
        
    Returns:
        재순위화된 검색 결과
    """
    try:
        url = "http://localhost/reranker/rerank"
        if top_k:
            url += f"?top_k={top_k}"
            
        payload = {
            "query": query,
            "results": passages,
            "total": len(passages),
            "reranked": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] 재순위화 중 오류: {e}")
        return {"results": passages}


# 통합 사용 예시
def search_with_reranking(query: str, top_k: int = 5, rerank_top_k: int = 3) -> dict:
    """
    검색 후 결과 재순위화 수행
    
    Args:
        query: 검색 쿼리
        top_k: 검색할 결과 수
        rerank_top_k: 재순위화 후 반환할 결과 수
        
    Returns:
        재순위화된 검색 결과
    """
    # 1. RAG 검색 수행
    search_results = search_rag(query, top_k=top_k)
    
    if not search_results or not search_results.get("results", []):
        print("[INFO] 검색 결과가 없습니다.")
        return {"query": query, "results": [], "total": 0}
    
    # 2. 검색 결과 재순위화
    passages = search_results.get("results", [])
    print(f"[INFO] {len(passages)}개 패시지 재순위화 중...")
    
    reranked_results = rerank_results(query, passages, top_k=rerank_top_k)
    
    if not reranked_results or not reranked_results.get("results", []):
        print("[WARNING] 재순위화 실패, 원본 결과 반환")
        return search_results
    
    print(f"[INFO] 재순위화 완료: {len(reranked_results.get('results', []))}개 결과")
    
    return reranked_results


def get_document_with_reranking(doc_id: str, use_title_as_query: bool = True) -> dict:
    """
    문서 검색 및 패시지 재순위화
    
    Args:
        doc_id: 문서 ID
        use_title_as_query: 문서 제목을 쿼리로 사용할지 여부
        
    Returns:
        패시지가 재순위화된 문서 정보
    """
    # 1. 문서 검색
    doc_info = get_document(doc_id)
    
    if not doc_info:
        print(f"[INFO] 문서 ID '{doc_id}'에 대한 정보를 찾을 수 없습니다.")
        return {}
    
    # 2. 패시지 추출
    passages = doc_info.get("passages", [])
    
    if not passages:
        print("[INFO] 재순위화할 패시지가 없습니다.")
        return doc_info
    
    # 3. 쿼리 생성
    query = ""
    if use_title_as_query:
        query = doc_info.get("title", "")
    
    if not query:
        query = doc_id  # 문서 ID를 쿼리로 사용
    
    print(f"[INFO] {len(passages)}개 패시지 재순위화 중 (쿼리: '{query}')")
    
    # 4. 패시지 재순위화
    reranked_results = rerank_results(query, passages)
    
    if not reranked_results or not reranked_results.get("results", []):
        print("[WARNING] 재순위화 실패, 원본 결과 반환")
        return doc_info
    
    # 5. 문서 정보 업데이트
    doc_info["passages"] = reranked_results.get("results", [])
    doc_info["reranked"] = True
    
    print(f"[INFO] 패시지 재순위화 완료")
    
    return doc_info


if __name__ == "__main__":
    # 예제 1: 검색 및 재순위화
    print("\n=== 예제 1: 검색 및 재순위화 ===")
    query = "메타버스 기술의 발전"
    results = search_with_reranking(query, top_k=10, rerank_top_k=3)
    
    # 결과 출력
    if results and results.get("results", []):
        print(f"\n쿼리: {query}")
        print(f"재순위화된 결과 ({len(results['results'])}개):")
        for i, result in enumerate(results["results"]):
            print(f"\n--- 결과 #{i+1} (점수: {result.get('rerank_score', 0):.4f}) ---")
            print(f"제목: {result.get('title', '제목 없음')}")
            print(f"텍스트: {result.get('text', '').strip()[:200]}...")
    
    # 예제 2: 문서 패시지 재순위화
    print("\n=== 예제 2: 문서 패시지 재순위화 ===")
    doc_id = "20220804-메타버스 뉴스"  # 또는 해시값 사용
    doc_info = get_document_with_reranking(doc_id)
    
    # 결과 출력
    if doc_info and doc_info.get("passages", []):
        print(f"\n문서: {doc_info.get('title', doc_id)}")
        print(f"재순위화된 패시지 ({len(doc_info['passages'])}개):")
        for i, passage in enumerate(doc_info['passages'][:3]):
            print(f"\n{i+1}. (점수: {passage.get('rerank_score', 0):.4f})")
            print(f"텍스트: {passage.get('text', '')[:200]}...") 