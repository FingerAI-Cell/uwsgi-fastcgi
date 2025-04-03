"""
Reranker FastCGI application
"""

import os
import json
from typing import Dict, List, Any, Optional
from flask import Flask, request, Response
from pydantic import BaseModel, Field

# 상대 경로 import 대신 절대 경로 import로 변경
from service import RerankerService

# 데이터 모델 정의
class PassageModel(BaseModel):
    """Single passage model"""
    passage_id: Optional[Any] = None
    doc_id: Optional[str] = None
    text: str
    score: Optional[float] = None
    position: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchResultModel(BaseModel):
    """Search result containing multiple passages"""
    query: str
    results: List[PassageModel]
    total: Optional[int] = None
    reranked: Optional[bool] = False


class RerankerResponseModel(BaseModel):
    """Response model for reranker API"""
    query: str
    results: List[PassageModel]
    total: int
    reranked: bool = True


# Flask 앱 생성
app = Flask(__name__)

# 서비스 인스턴스 생성
reranker_service = None


def get_reranker_service():
    """Get reranker service instance"""
    global reranker_service
    if reranker_service is None:
        config_path = os.environ.get("RERANKER_CONFIG", "/reranker/config.json")
        reranker_service = RerankerService(config_path)
    return reranker_service


@app.route("/reranker/health")
def health_check():
    """Health check endpoint"""
    return Response(
        json.dumps({"status": "ok", "service": "reranker"}, ensure_ascii=False),
        mimetype='application/json; charset=utf-8'
    )


@app.route("/reranker/rerank", methods=["POST"])
def rerank_passages():
    """
    Rerank search results based on query relevance
    
    Returns:
        Reranked search results
    """
    try:
        data = request.get_json()
        top_k = request.args.get("top_k", type=int)
        
        # Convert to SearchResultModel
        search_result = SearchResultModel(**data)
        
        # Process search results
        reranked = get_reranker_service().process_search_results(
            search_result.query, 
            search_result.dict(), 
            top_k
        )
        
        # Set total value
        reranked["total"] = len(reranked["results"])
        
        return Response(
            json.dumps(reranked, ensure_ascii=False),
            mimetype='application/json; charset=utf-8'
        )
        
    except Exception as e:
        return Response(
            json.dumps({"error": f"Reranking failed: {str(e)}"}, ensure_ascii=False),
            status=500,
            mimetype='application/json; charset=utf-8'
        )


@app.route("/reranker/batch_rerank", methods=["POST"])
def batch_rerank():
    """
    Batch rerank multiple queries and their passages
    
    Returns:
        List of reranked results for each query
    """
    try:
        data = request.get_json()
        top_k = request.args.get("top_k", type=int)
        
        # Process each query
        results = []
        for query_data in data:
            search_result = SearchResultModel(**query_data)
            reranked = get_reranker_service().process_search_results(
                search_result.query,
                search_result.dict(),
                top_k
            )
            results.append(reranked)
        
        return Response(
            json.dumps(results, ensure_ascii=False),
            mimetype='application/json; charset=utf-8'
        )
        
    except Exception as e:
        return Response(
            json.dumps({"error": f"Batch reranking failed: {str(e)}"}, ensure_ascii=False),
            status=500,
            mimetype='application/json; charset=utf-8'
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000) 