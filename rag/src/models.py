from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import TextStreamer, GenerationConfig
from openai import OpenAI
from abc import ABC, abstractmethod
import numpy as np
import torch
import warnings
import os
import logging
import time

# 멀티프로세싱 관련 설정
import torch.multiprocessing as mp
# spawn 메소드 사용 (fork 대신)
mp.set_start_method('spawn', force=True)

# 특정 경고 메시지 무시
# warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

class Model:
    def __init__(self, config):
        self.config = config 
        self.set_gpu()
       
    def set_gpu(self):
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 사용
            self.device = torch.device("cuda:0")
            # CUDA 초기화 확인
            torch.cuda.init()
            # 현재 CUDA 디바이스 메모리 비우기
            torch.cuda.empty_cache()
            # CUDA 상태 확인
            logging.info(f"GPU is available. Using device: {self.device}")
            logging.info(f"CUDA initialized: {torch.cuda.is_initialized()}")
            logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logging.info(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            logging.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
        else:
            self.device = torch.device("cpu")
            logging.info("GPU is not available. Using CPU instead.")

    def set_random_state(self, seed=42):
        self.random_state = seed


class EmbModel(Model):
    def __init__(self, config):
        super().__init__(config)
        # GPU/CPU에 따른 기본 배치 사이즈 설정
        self.default_batch_sizes = {
            "cpu": 12,
            "gpu": 256
        }
    
    def set_embbeding_config(self, batch_size=None, max_length=1024):
        # GPU 여부에 따라 기본 배치 사이즈 선택
        if batch_size is None:
            mode = "gpu" if torch.cuda.is_available() else "cpu"
            batch_size = self.default_batch_sizes[mode]
            logging.info(f"Using {mode.upper()} mode with batch_size: {batch_size}")
        
        self.emb_config = {
            "batch_size": batch_size,
            "max_length": max_length
        }

    def set_emb_model(self, model_type='bge'):
        if model_type == 'bge':
            from FlagEmbedding import BGEM3FlagModel
            mode = "gpu" if torch.cuda.is_available() else "cpu"
            batch_size = self.default_batch_sizes[mode]
            
            model_path = os.getenv('MODEL_PATH', '/rag/models/bge-m3')  # 환경 변수에서 경로 가져오기
            if not os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
                logging.error(f"모델 파일이 {model_path}에 존재하지 않습니다. setup.sh를 실행하여 모델을 먼저 다운로드해주세요.")
                raise FileNotFoundError(f"Model files not found in {model_path}")
                
            self.bge_emb = BGEM3FlagModel(
                model_path,  # 환경 변수에서 가져온 경로 사용
                use_fp16=True,
                device=self.device,
                compute_dtype=torch.float16,
                batch_size=batch_size  # 모델 초기화 시 배치 사이즈 설정
            )
            logging.info(f"Loaded BGE model from {model_path} on device: {self.device} with batch_size: {batch_size}")
            if torch.cuda.is_available():
                logging.info(f"GPU Memory after model load: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        
             
    def bge_embed_data(self, text):
        """
        BGE 모델을 사용하여 텍스트를 임베딩합니다.
        오류 발생 시 CPU로 폴백하는 안전한 임베딩 처리를 제공합니다.
        """
        start_time = time.time()
        print(f"[TIMING] 임베딩 시작 - 텍스트 길이: {len(text)}")
        
        # 메모리 상태 확인
        if torch.cuda.is_available():
            try:
                logging.info(f"GPU Memory before embedding: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                logging.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
                # 메모리 정리
                torch.cuda.empty_cache()
                logging.info(f"GPU Memory after cache clear: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            except Exception as e:
                logging.error(f"Error checking GPU memory: {str(e)}")
        
        # 임베딩 수행
        batch_size = self.emb_config['batch_size']
        logging.info(f"Using batch size: {batch_size}")
        use_cpu_fallback = False
        
        try:
            # GPU 임베딩 시도
            if isinstance(text, str):
                with torch.no_grad():  # 메모리 사용량 줄이기
                    embeddings = self.bge_emb.encode(text, max_length=self.emb_config['max_length'])['dense_vecs']
                logging.info(f"Embedded single text with batch size {batch_size}")
            else:
                total_samples = len(text)
                logging.info(f"Embedding {total_samples} texts with batch size {batch_size}")
                
                # 메모리 효율을 위한 배치 처리
                embeddings = []
                for i in range(0, total_samples, batch_size):
                    with torch.no_grad():  # 메모리 사용량 줄이기
                        batch_texts = list(text[i:i + batch_size])
                        batch_embeddings = self.bge_emb.encode(batch_texts, batch_size=batch_size, max_length=self.emb_config['max_length'])['dense_vecs']
                    embeddings.extend(batch_embeddings)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # 배치 처리 후 메모리 정리
                
                logging.info(f"Completed embedding {total_samples} texts in {total_samples//batch_size + (1 if total_samples%batch_size else 0)} batches")
            
            end_time = time.time()
            print(f"[TIMING] 임베딩 완료: {(end_time - start_time):.4f}초")
            
        except RuntimeError as e:
            # OOM 오류 발생 시 배치 사이즈 감소 또는 CPU 폴백
            if "out of memory" in str(e) or "CUDA" in str(e):
                error_time = time.time()
                logging.warning(f"GPU error during embedding: {str(e)}")
                print(f"[TIMING] GPU 오류 발생: {(error_time - start_time):.4f}초")
                
                # 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # CPU 폴백 - 심각한 CUDA 오류 발생 시
                if "CUDA" in str(e) or ("out of memory" in str(e) and batch_size <= 32):
                    logging.warning("Falling back to CPU for embedding")
                    print(f"[TIMING] CPU 폴백으로 임베딩 재시도")
                    
                    # 임시로 디바이스를 CPU로 변경
                    original_device = self.bge_emb.device
                    self.bge_emb.device = torch.device("cpu")
                    use_cpu_fallback = True
                    
                    try:
                        if isinstance(text, str):
                            embeddings = self.bge_emb.encode(text, max_length=self.emb_config['max_length'])['dense_vecs']
                        else:
                            embeddings = []
                            for i in range(0, len(text), 8):  # CPU에서는 더 작은 배치 사이즈
                                batch_texts = list(text[i:i + 8])
                                batch_embeddings = self.bge_emb.encode(batch_texts, batch_size=8, max_length=self.emb_config['max_length'])['dense_vecs']
                                embeddings.extend(batch_embeddings)
                        
                        cpu_end_time = time.time()
                        print(f"[TIMING] CPU 임베딩 완료: {(cpu_end_time - error_time):.4f}초")
                        # 디바이스 원복
                        self.bge_emb.device = original_device
                        
                    except Exception as cpu_e:
                        logging.error(f"CPU embedding also failed: {str(cpu_e)}")
                        # 디바이스 원복 후 예외 발생
                        self.bge_emb.device = original_device
                        raise
                
                # OOM일 경우 배치 사이즈 감소 시도
                else:
                    original_batch_size = batch_size
                    batch_size = batch_size // 2
                    logging.warning(f"GPU OOM detected. Reducing batch size from {original_batch_size} to {batch_size}")
                    
                    # 재시도
                    retry_start = time.time()
                    print(f"[TIMING] 임베딩 재시도 (배치 크기 감소)")
                    
                    if isinstance(text, str):
                        with torch.no_grad():
                            embeddings = self.bge_emb.encode(text, batch_size=batch_size, max_length=self.emb_config['max_length'])['dense_vecs']
                    else:
                        embeddings = []
                        for i in range(0, len(text), batch_size):
                            with torch.no_grad():
                                batch_texts = list(text[i:i + batch_size])
                                batch_embeddings = self.bge_emb.encode(batch_texts, batch_size=batch_size, max_length=self.emb_config['max_length'])['dense_vecs']
                            embeddings.extend(batch_embeddings)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    
                    retry_end = time.time()
                    print(f"[TIMING] 임베딩 재시도 완료: {(retry_end - retry_start):.4f}초")
                
                print(f"[TIMING] 총 임베딩 시간 (오류 처리 포함): {(time.time() - start_time):.4f}초")
            
            else:
                # 다른 런타임 오류 처리
                error_time = time.time()
                print(f"[TIMING] 임베딩 오류: {(error_time - start_time):.4f}초, 오류: {str(e)}")
                raise
        
        except Exception as e:
            # 기타 예외 처리
            error_time = time.time()
            logging.error(f"Unhandled exception during embedding: {str(e)}")
            print(f"[TIMING] 예상치 못한 오류: {(error_time - start_time):.4f}초, 오류: {str(e)}")
            # 간단한 임베딩으로 대체 (텍스트 길이가 짧으면)
            if isinstance(text, str) and len(text) < 10:
                logging.warning("Using fallback random embedding for short text")
                return np.random.rand(1024).astype(np.float32).tolist()
            raise
        
        # 메모리 상태 확인 (임베딩 후)
        if torch.cuda.is_available() and not use_cpu_fallback:
            try:
                logging.info(f"GPU Memory after embedding: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                torch.cuda.empty_cache()
            except Exception as e:
                logging.error(f"Error checking GPU memory after embedding: {str(e)}")
        
        # float32로 변환 (필요 시)
        try:
            if isinstance(embeddings[0], np.ndarray):
                return [e.astype(np.float32) for e in embeddings]
            return embeddings
        except:
            return embeddings

    def calc_emb_similarity(self, emb1, emb2, metric='L2'):
        if metric == 'L2':   # Euclidean distance
            l2_distance = np.linalg.norm(emb1 - emb2)
            return l2_distance

    @abstractmethod
    def get_hf_encoder(self):
        pass

    @abstractmethod 
    def get_cohere_encoder(self, cohere_api):
        pass


class LLMOpenAI(Model):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI()

    def set_generation_config(self, max_tokens=500, temperature=0.9):
        self.gen_config = {
            "max_tokens": max_tokens,
            "temperature": temperature
        }

    def get_response(self, query, role="너는 금융권에서 일하고 있는 조수로, 회사 규정에 대해 알려주는 역할을 맡고 있어. 사용자 질문에 대해 간단 명료하게 답을 해줘.", model='gpt-4'):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": query},
                ],
                max_tokens=self.gen_config['max_tokens'],
                temperature=self.gen_config['temperature'],
            )    
        except Exception as e:
            return f"Error: {str(e)}"
        return response.choices[0].message.content

    def set_prompt_template(self, query, context):
        self.rag_prompt_template = """
        다음 질문에 대해 주어진 정보를 참고해서 답을 해줘.
        주어진 정보: {context}
        --------------------------------
        질문: {query} 
        """
        return self.rag_prompt_template.format(query=query, context=context)

