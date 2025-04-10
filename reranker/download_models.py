"""
FlashRank 모델을 로컬에 다운로드하는 스크립트
"""

import os
import json
import argparse
import logging
from typing import Optional
import subprocess
import sys


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """설정 파일을 불러옵니다."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류 발생: {e}")
        return {
            "reranker_type": "flashrank",
            "model_name": "intfloat/e5-base-v2",
            "model_dir": "./models"
        }


def ensure_flashrank_installed():
    """FlashRank가 설치되어 있는지 확인하고, 없으면 설치합니다."""
    try:
        import flashrank
        logger.info(f"FlashRank가 설치되어 있습니다. 버전: {getattr(flashrank, '__version__', '알 수 없음')}")
        return True
    except ImportError:
        logger.warning("FlashRank가 설치되어 있지 않습니다. 설치를 시도합니다.")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/AnswerDotAI/flashrank.git"
            ])
            logger.info("FlashRank가 성공적으로 설치되었습니다.")
            return True
        except Exception as e:
            logger.error(f"FlashRank 설치 중 오류 발생: {e}")
            return False


def download_model(model_name: str, model_dir: str, force: bool = False) -> Optional[str]:
    """
    FlashRank를 사용하여 모델을 다운로드합니다.
    
    Args:
        model_name: 모델 이름 또는 경로 (Hugging Face Hub)
        model_dir: 모델 저장 디렉토리
        force: 이미 있는 모델도 강제로 다시 다운로드
        
    Returns:
        다운로드된 모델 경로 또는 None
    """
    # 모델 디렉토리 생성
    os.makedirs(model_dir, exist_ok=True)
    
    # 모델 이름에서 저장 경로 구성
    model_path = os.path.join(model_dir, os.path.basename(model_name))
    
    # 모델이 이미 존재하고 force가 False라면 다운로드 건너뛰기
    if os.path.exists(model_path) and not force:
        logger.info(f"모델이 이미 존재합니다: {model_path}")
        return model_path
    
    # FlashRank가 있는지 확인
    if not ensure_flashrank_installed():
        logger.error("FlashRank를 설치할 수 없어 모델 다운로드를 계속할 수 없습니다.")
        return None
        
    try:
        logger.info(f"모델 다운로드 중: {model_name}")
        
        # FlashRank를 사용하여 모델 다운로드
        import flashrank
        
        # 다양한 API 시도
        try:
            # 방법 1: download_model 함수가 있는 경우
            if hasattr(flashrank, 'download_model'):
                logger.info("flashrank.download_model 함수 사용")
                model_path = flashrank.download_model(model_name, output_dir=model_path)
            # 방법 2: Reranker 클래스 생성자에서 자동 다운로드
            elif hasattr(flashrank, 'Reranker'):
                logger.info("flashrank.Reranker 클래스 사용")
                reranker = flashrank.Reranker(model_name_or_path=model_name)
                model_path = reranker.model_path if hasattr(reranker, 'model_path') else model_path
            # 방법 3: transformers 사용
            else:
                logger.info("transformers 라이브러리 사용")
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=model_path)
                model_path = tokenizer.save_pretrained(model_path) or model_path
                model.save_pretrained(model_path)
        except Exception as e:
            logger.error(f"FlashRank API 사용 중 오류: {e}")
            logger.info("transformers 라이브러리로 대체")
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from huggingface_hub import snapshot_download
            
            # Hugging Face Hub에서 모델 다운로드
            model_path = snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            
            # 모델과 토크나이저 로드하여 테스트
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        logger.info(f"모델 {model_name} 다운로드 완료: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"모델 다운로드 중 오류 발생: {e}")
        return None


def test_flashrank_model(model_path: str):
    """FlashRank 모델을 테스트합니다."""
    try:
        logger.info("FlashRank 모델 테스트 중...")
        import flashrank
        
        # 다양한 API 시도
        try:
            if hasattr(flashrank, 'Reranker'):
                reranker = flashrank.Reranker(model_name_or_path=model_path)
                scores = reranker.compute_score(
                    query="테스트 쿼리", 
                    passages=["테스트 문서입니다.", "이것은 테스트용 패시지입니다."]
                )
                logger.info(f"FlashRank 테스트 성공! Reranker.compute_score 점수: {scores}")
            elif hasattr(flashrank, 'CrossEncoder'):
                reranker = flashrank.CrossEncoder(model_name_or_path=model_path)
                scores = reranker.rerank(
                    query="테스트 쿼리", 
                    passages=["테스트 문서입니다.", "이것은 테스트용 패시지입니다."]
                )
                logger.info(f"FlashRank 테스트 성공! CrossEncoder.rerank 결과: {scores}")
            elif hasattr(flashrank, 'rerank'):
                scores = flashrank.rerank(
                    query="테스트 쿼리", 
                    passages=["테스트 문서입니다.", "이것은 테스트용 패시지입니다."],
                    model=model_path
                )
                logger.info(f"FlashRank 테스트 성공! flashrank.rerank 결과: {scores}")
            else:
                # transformers로 대체 테스트
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch
                
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                
                # 간단한 테스트
                test_inputs = tokenizer([("테스트 쿼리", "테스트 문서입니다.")], 
                                        padding=True, 
                                        truncation=True, 
                                        return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**test_inputs)
                    scores = outputs.logits
                
                logger.info(f"Transformers 테스트 성공! 결과 형태: {scores.shape}")
        except Exception as e:
            logger.error(f"FlashRank API 테스트 중 오류: {e}")
            # transformers로 대체 테스트
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # 간단한 테스트
            test_inputs = tokenizer([("테스트 쿼리", "테스트 문서입니다.")], 
                                    padding=True, 
                                    truncation=True, 
                                    return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**test_inputs)
                scores = outputs.logits
            
            logger.info(f"Transformers 테스트 성공! 결과 형태: {scores.shape}")
            
    except Exception as e:
        logger.error(f"모델 테스트 중 오류 발생: {e}")


def download_reranker_model():
    """FlashRank 모델 다운로드"""
    # 설정 로드
    config_path = os.environ.get("RERANKER_CONFIG", "/reranker/config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
    config = load_config(config_path)
    
    model_name = config.get("model_name", "intfloat/e5-base-v2")
    model_dir = config.get("model_dir", "/reranker/models")
    
    logger.info(f"FlashRank 모델 다운로드 시작: {model_name}")
    model_path = download_model(model_name, model_dir)
    
    if model_path:
        # 모델 테스트
        test_flashrank_model(model_path)


def main():
    parser = argparse.ArgumentParser(description="FlashRank 모델 다운로드")
    parser.add_argument("--model", type=str, default=None, help="다운로드할 모델 이름")
    parser.add_argument("--dir", type=str, default=None, help="모델 저장 디렉토리")
    parser.add_argument("--force", action="store_true", help="이미 있는 모델도 강제로 다시 다운로드")
    
    args = parser.parse_args()
    
    if args.model and args.dir:
        # 특정 모델 다운로드
        model_path = download_model(args.model, args.dir, args.force)
        if model_path:
            test_flashrank_model(model_path)
    else:
        # 설정에서 모델 다운로드
        download_reranker_model()


if __name__ == "__main__":
    main() 