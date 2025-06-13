#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRC 모듈 로딩 테스트 스크립트
"""

import os
import sys
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mrc_import():
    """MRC 모듈 임포트 테스트"""
    try:
        logger.info("Python 경로:")
        for path in sys.path:
            logger.info(f"  {path}")
            
        logger.info("현재 디렉토리: %s", os.getcwd())
        
        # src 디렉토리 존재 확인
        if os.path.exists('src'):
            logger.info("src 디렉토리 존재함")
            if os.path.exists('src/mrc'):
                logger.info("src/mrc 디렉토리 존재함")
                logger.info("src/mrc 디렉토리 내용:")
                for item in os.listdir('src/mrc'):
                    logger.info(f"  {item}")
        
        # MRC 모듈 임포트 시도
        logger.info("MRC 모듈 임포트 시도...")
        try:
            from src.mrc import MRCReranker
            logger.info("MRC 모듈 임포트 성공 (from src.mrc)")
            return True
        except ImportError as e:
            logger.error(f"MRC 모듈 임포트 실패 (from src.mrc): {e}")
            
            # 다른 경로 시도
            try:
                import src.mrc
                logger.info("src.mrc 모듈 임포트 성공")
                logger.info(f"src.mrc 모듈 내용: {dir(src.mrc)}")
                return True
            except ImportError as e:
                logger.error(f"src.mrc 모듈 임포트 실패: {e}")
                
            # 상대 경로 시도
            try:
                from .src.mrc import MRCReranker
                logger.info("MRC 모듈 임포트 성공 (from .src.mrc)")
                return True
            except (ImportError, ValueError) as e:
                logger.error(f"MRC 모듈 임포트 실패 (from .src.mrc): {e}")
                
            # 절대 경로 시도
            try:
                from reranker.src.mrc import MRCReranker
                logger.info("MRC 모듈 임포트 성공 (from reranker.src.mrc)")
                return True
            except ImportError as e:
                logger.error(f"MRC 모듈 임포트 실패 (from reranker.src.mrc): {e}")
                
            return False
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        return False

def test_pytorch_import():
    """PyTorch 관련 라이브러리 임포트 테스트"""
    try:
        import torch
        logger.info(f"PyTorch 버전: {torch.__version__}")
        
        import torchtext
        logger.info(f"TorchText 버전: {torchtext.__version__}")
        
        import pytorch_lightning
        logger.info(f"PyTorch Lightning 버전: {pytorch_lightning.__version__}")
        
        return True
    except ImportError as e:
        logger.error(f"PyTorch 관련 라이브러리 임포트 실패: {e}")
        return False

def test_munch_import():
    """Munch 라이브러리 임포트 테스트"""
    try:
        import munch
        logger.info(f"Munch 버전: {munch.__version__ if hasattr(munch, '__version__') else 'unknown'}")
        
        # munchify 함수 테스트
        test_dict = {"a": 1, "b": 2}
        munchified = munch.munchify(test_dict)
        logger.info(f"munchify 테스트: {munchified.a}, {munchified.b}")
        
        return True
    except ImportError as e:
        logger.error(f"Munch 라이브러리 임포트 실패: {e}")
        return False

def test_mrc_files():
    """MRC 모듈 파일 존재 여부 확인"""
    try:
        # 설정 파일 경로
        config_paths = [
            "/reranker/src/mrc/mrc.json",
            "src/mrc/mrc.json"
        ]
        
        # 모델 파일 경로
        model_paths = [
            "/reranker/src/mrc/mrc.ckpt",
            "src/mrc/mrc.ckpt"
        ]
        
        # 설정 파일 확인
        for path in config_paths:
            exists = os.path.exists(path)
            logger.info(f"설정 파일 '{path}' 존재 여부: {exists}")
            
            if exists:
                try:
                    import json
                    with open(path, 'r') as f:
                        config = json.load(f)
                    logger.info(f"설정 파일 내용: {config}")
                except Exception as e:
                    logger.error(f"설정 파일 읽기 실패: {e}")
        
        # 모델 파일 확인
        for path in model_paths:
            exists = os.path.exists(path)
            logger.info(f"모델 파일 '{path}' 존재 여부: {exists}")
            
            if exists:
                logger.info(f"모델 파일 크기: {os.path.getsize(path)} 바이트")
        
        return True
    except Exception as e:
        logger.error(f"파일 확인 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    logger.info("===== MRC 모듈 테스트 시작 =====")
    
    test_pytorch_import()
    test_munch_import()
    test_mrc_files()
    success = test_mrc_import()
    
    logger.info(f"===== MRC 모듈 테스트 완료: {'성공' if success else '실패'} =====")
    sys.exit(0 if success else 1) 