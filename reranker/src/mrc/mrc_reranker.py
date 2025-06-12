import logging
from typing import List, Dict, Any, Optional, Union, Tuple

from .controller import MRCController

logger = logging.getLogger(__name__)

class MRCReranker:
    """MRC 결과를 활용한 재랭킹 기능을 제공하는 클래스"""
    
    _instance = None  # 싱글톤 인스턴스
    
    @classmethod
    def get_instance(cls, config_path=None, model_path=None):
        """싱글톤 패턴으로 인스턴스 반환"""
        if cls._instance is None:
            logger.info(f"Creating new MRCReranker instance")
            cls._instance = cls(config_path, model_path)
        else:
            logger.info("Returning existing MRCReranker instance")
        return cls._instance
    
    def __init__(self, config_path=None, model_path=None):
        """
        MRC 기반 재랭커 초기화
        
        Args:
            config_path: MRC 모델 설정 경로
            model_path: MRC 모델 체크포인트 경로
        """
        self.mrc_controller = MRCController.get_instance(config_path, model_path)
        logger.info("MRCReranker initialized")
        
    def rerank(self, query: str, passages: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        MRC 기반 재랭킹 수행
        
        Args:
            query: 검색 쿼리
            passages: 재랭킹할 패시지 목록
            top_k: 반환할 상위 결과 수
            
        Returns:
            재랭킹된 패시지 목록
        """
        logger.info(f"MRC 기반 재랭킹 시작: query='{query}', passages={len(passages)}")
        
        # MRC 입력 생성
        samples = []
        for i, passage in enumerate(passages):
            samples.append({
                'question': query,
                'context': passage.get('text', ''),
                'temperature': 1.0,
                'original_index': i  # 원본 인덱스 추적
            })
        
        # MRC 추론 수행
        mrc_results = self.mrc_controller.infer_multi(samples)
        
        # 결과 연결 및 점수 업데이트
        for i, (passage, mrc_result) in enumerate(zip(passages, mrc_results)):
            # 원본 점수 보존
            original_score = passage.get('score', 0.0)
            
            # MRC 결과 저장
            passage['mrc_answer'] = mrc_result['answer']
            passage['mrc_char_ids'] = mrc_result['char_ids']
            passage['mrc_score'] = mrc_result['answerability']
            
            # 최종 점수 계산 (원본 점수와 MRC 점수의 조합)
            passage['final_score'] = original_score * 0.3 + mrc_result['answerability'] * 0.7
            
            # 디버그 로깅
            logger.debug(f"Passage {i}: original_score={original_score:.4f}, mrc_score={mrc_result['answerability']:.4f}, final_score={passage['final_score']:.4f}")
        
        # 최종 점수로 정렬
        reranked_passages = sorted(passages, key=lambda x: x.get('final_score', 0), reverse=True)
        
        # top_k 적용
        if top_k and isinstance(top_k, int) and top_k > 0:
            reranked_passages = reranked_passages[:top_k]
            
        logger.info(f"MRC 기반 재랭킹 완료: {len(reranked_passages)} 결과 반환")
        return reranked_passages
    
    def process_search_results(self, query: str, search_result: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """
        검색 결과에 MRC 기반 재랭킹 적용
        (RerankerService.process_search_results와 동일한 인터페이스)
        
        Args:
            query: 검색 쿼리
            search_result: 검색 결과 딕셔너리
            top_k: 반환할 상위 결과 수
            
        Returns:
            재랭킹된 검색 결과
        """
        # 재랭킹 수행
        passages = search_result.get("results", [])
        reranked_passages = self.rerank(query, passages, top_k)
        
        # 결과 포맷팅
        result = {
            "query": query,
            "results": reranked_passages,
            "total": len(reranked_passages),
            "reranked": True,
            "reranker_type": "mrc"
        }
        
        return result
        
    def hybrid_rerank(self, query: str, passages: List[Dict[str, Any]], 
                      flashrank_scores: List[float], 
                      weight_mrc: float = 0.7,
                      top_k: Optional[int] = None,
                      return_mrc_scores: bool = False) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[float]]]:
        """
        FlashRank 결과와 MRC 결과를 조합한 하이브리드 재랭킹
        
        Args:
            query: 검색 쿼리
            passages: 재랭킹할 패시지 목록
            flashrank_scores: FlashRank에서 계산한 점수 목록
            weight_mrc: MRC 점수 가중치 (0~1 사이)
            top_k: 반환할 상위 결과 수
            return_mrc_scores: MRC 점수 목록도 함께 반환할지 여부
            
        Returns:
            하이브리드 재랭킹된 패시지 목록, 또는 (패시지 목록, MRC 점수 목록) 튜플
        """
        logger.info(f"하이브리드 재랭킹 시작: query='{query}', passages={len(passages)}, weight_mrc={weight_mrc}")
        
        # MRC 입력 생성 및 추론
        samples = []
        for passage in passages:
            samples.append({
                'question': query,
                'context': passage.get('text', ''),
                'temperature': 1.0
            })
        
        mrc_results = self.mrc_controller.infer_multi(samples)
        mrc_scores = []  # MRC 점수 목록 저장
        
        # 점수 결합 및 결과 업데이트
        weight_flashrank = 1.0 - weight_mrc
        for i, (passage, mrc_result, flashrank_score) in enumerate(zip(passages, mrc_results, flashrank_scores)):
            # MRC 점수 저장
            mrc_score = mrc_result['answerability']
            mrc_scores.append(mrc_score)
            
            # MRC 결과 저장
            passage['mrc_answer'] = mrc_result['answer']
            passage['mrc_char_ids'] = mrc_result['char_ids']
            passage['mrc_score'] = mrc_score
            passage['flashrank_score'] = flashrank_score
            
            # 하이브리드 점수 계산
            passage['hybrid_score'] = (flashrank_score * weight_flashrank) + (mrc_score * weight_mrc)
            passage['score'] = passage['hybrid_score']  # 기본 score 필드 업데이트
            
            logger.debug(f"Passage {i}: flashrank={flashrank_score:.4f}, mrc={mrc_score:.4f}, hybrid={passage['hybrid_score']:.4f}")
        
        # 하이브리드 점수로 정렬
        reranked_passages = sorted(passages, key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        # top_k 적용 (점수 목록도 함께 정렬)
        if top_k and isinstance(top_k, int) and top_k > 0:
            # 인덱스와 함께 정렬된 패시지 목록 생성
            indexed_passages = [(i, p) for i, p in enumerate(reranked_passages)]
            # top_k까지만 선택
            top_indexed_passages = indexed_passages[:top_k]
            # 인덱스와 패시지 분리
            indices, reranked_passages = zip(*top_indexed_passages) if top_indexed_passages else ([], [])
            # 같은 순서로 mrc_scores 재정렬 (필요한 경우)
            if return_mrc_scores:
                mrc_scores = [mrc_scores[i] for i in indices]
                
            # 리스트로 변환 (zip 결과는 튜플)
            reranked_passages = list(reranked_passages)
            
        logger.info(f"하이브리드 재랭킹 완료: {len(reranked_passages)} 결과 반환")
        
        if return_mrc_scores:
            return reranked_passages, mrc_scores
        else:
            return reranked_passages 