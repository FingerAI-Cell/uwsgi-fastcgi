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
import threading

# 멀티프로세싱 관련 설정 - 서버에서는 이미 프로세스가 생성된 후에만 이 코드가 실행되므로 
# spawn 메소드를 설정하면 오류가 발생할 수 있음
# 대신 메인 프로세스에서만 GPU 관련 작업을 처리하도록 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU 순서를 일관되게 유지
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 명시적으로 첫 번째 GPU만 사용

# CUDA 초기화 관련 설정
torch.backends.cudnn.benchmark = True           # 성능 향상
torch.backends.cudnn.deterministic = False      # 성능 우선
torch.backends.cuda.matmul.allow_tf32 = True    # TensorFloat-32 활성화 (Ampere 이상)
torch.backends.cudnn.allow_tf32 = True          # TF32 연산 허용

class Model:
    def __init__(self, config):
        self.config = config 
        self.set_gpu()
        self.gpu_initialized = False  # GPU 초기화 상태 추적
       
    def set_gpu(self):
        try:
            if torch.cuda.is_available():
                # CUDA 초기화 전 메모리 정리
                torch.cuda.empty_cache()
                
                # CUDA 초기화 - 첫 사용 시 발생하는 지연 미리 처리
                torch.cuda.init()
                _ = torch.zeros(1).cuda()  # 첫 CUDA 할당 미리 수행
                
                self.device = torch.device("cuda:0")
                self.gpu_initialized = True
                
                # CUDA 상태 출력
                device_name = torch.cuda.get_device_name(0)
                device_props = torch.cuda.get_device_properties(0)
                logging.info(f"GPU initialized: {device_name}")
                logging.info(f"GPU total memory: {device_props.total_memory/1024**2:.2f}MB")
                logging.info(f"GPU compute capability: {device_props.major}.{device_props.minor}")
                logging.info(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                logging.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
            else:
                logging.warning("No CUDA device available, using CPU")
                self.device = torch.device("cpu")
        except Exception as e:
            logging.error(f"Error initializing GPU: {str(e)}")
            logging.warning("Falling back to CPU due to GPU initialization error")
            self.device = torch.device("cpu")

    def set_random_state(self, seed=42):
        self.random_state = seed


class EmbModel(Model):
    def __init__(self, config):
        super().__init__(config)
        # GPU/CPU에 따른 기본 배치 사이즈 설정
        self.default_batch_sizes = {
            "cpu": 12,
            "gpu": 64  # 더 작은 배치 사이즈로 설정, 안정성 향상
        }
        self._model_loaded = False  # 모델 로드 상태 추적
        # GPU 동시 접근 제한 (환경변수로 설정 가능)
        # MAX_GPU_WORKERS 환경변수로 GPU 동시 접근 수 제한
        self.max_gpu_workers = int(os.getenv('MAX_GPU_WORKERS', '50'))  # pipe.py와 통일하여 50으로 설정
        # 세마포어는 초기화하지 않고 get_gpu_semaphore 메서드에서 획득
        # 활성 작업 수 추적을 위한 변수
        self._active_gpu_tasks = 0
        self._task_lock = threading.Lock()
        logging.info(f"GPU 동시 작업 제한: {self.max_gpu_workers}개")
    
    def set_embbeding_config(self, batch_size=None, max_length=1024):
        # GPU 여부에 따라 기본 배치 사이즈 선택
        if batch_size is None:
            mode = "gpu" if torch.cuda.is_available() and self.gpu_initialized else "cpu"
            batch_size = self.default_batch_sizes[mode]
            logging.info(f"Using {mode.upper()} mode with batch_size: {batch_size}")
        
        self.emb_config = {
            "batch_size": batch_size,
            "max_length": max_length
        }

    def set_emb_model(self, model_type='bge'):
        if model_type == 'bge':
            from FlagEmbedding import BGEM3FlagModel
            
            # GPU 사용 가능 여부 재확인
            use_gpu = torch.cuda.is_available() and self.gpu_initialized
            device = self.device
            mode = "gpu" if use_gpu else "cpu"
            batch_size = self.default_batch_sizes[mode]
            
            model_path = os.getenv('MODEL_PATH', '/rag/models/bge-m3')
            if not os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
                logging.error(f"모델 파일이 {model_path}에 존재하지 않습니다.")
                raise FileNotFoundError(f"Model files not found in {model_path}")
            
            try:
                logging.info(f"Loading BGE model on {device} with batch_size {batch_size}")
                
                # 메모리 정리
                if use_gpu:
                    torch.cuda.empty_cache()
                
                # 모델 로드 전 메모리 상태 확인
                if use_gpu:
                    before_load = torch.cuda.memory_allocated()/1024**2
                    logging.info(f"GPU Memory before model load: {before_load:.2f}MB")
                
                # 모델 로드 시 타임아웃 설정 (안전장치)
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Model loading timed out")
                
                # 타임아웃 설정 (120초)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)
                
                try:
                    # 모델 생성
                    self.bge_emb = BGEM3FlagModel(
                        model_path,
                        use_fp16=use_gpu,
                        device=device,
                        compute_dtype=torch.float16 if use_gpu else torch.float32,
                        batch_size=batch_size
                    )
                    # 타임아웃 비활성화
                    signal.alarm(0)
                    self._model_loaded = True
                except TimeoutError:
                    logging.error("Model loading timed out, trying with smaller batch")
                    signal.alarm(0)
                    # 더 작은 배치 사이즈로 재시도
                    batch_size = max(1, batch_size // 4)
                    self.bge_emb = BGEM3FlagModel(
                        model_path,
                        use_fp16=use_gpu,
                        device=device,
                        compute_dtype=torch.float16 if use_gpu else torch.float32,
                        batch_size=batch_size
                    )
                    self._model_loaded = True
                
                # 모델 로드 후 메모리 상태 확인
                if use_gpu:
                    after_load = torch.cuda.memory_allocated()/1024**2
                    logging.info(f"GPU Memory after model load: {after_load:.2f}MB")
                    logging.info(f"Model size in memory: {after_load - before_load:.2f}MB")
                
                logging.info(f"Successfully loaded BGE model on {device}")
                
                # 모델 워밍업 (첫 추론 시간 단축)
                logging.info("Performing model warmup...")
                with torch.no_grad():
                    _ = self.bge_emb.encode("워밍업 텍스트", max_length=128)['dense_vecs']
                logging.info("Model warmup completed")
                
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                if use_gpu:
                    logging.warning("GPU error encountered. Trying with CPU...")
                    self.device = torch.device("cpu")
                    batch_size = self.default_batch_sizes["cpu"]
                    # CPU로 재시도
                    self.bge_emb = BGEM3FlagModel(
                        model_path,
                        use_fp16=False,
                        device=torch.device("cpu"),
                        compute_dtype=torch.float32,
                        batch_size=batch_size
                    )
                    self._model_loaded = True
                    logging.info(f"Successfully loaded BGE model on CPU as fallback")
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # 세마포어 관련 메서드 추가
    def get_gpu_semaphore(self):
        """GPU 접근을 제한하기 위한 세마포어를 반환합니다."""
        # InteractManager 클래스의 세마포어 사용 (모든 임베딩 처리에서 공유)
        from .pipe import InteractManager
        return InteractManager.get_gpu_semaphore()
            
    def bge_embed_data(self, text):
        """
        BGE 모델을 사용하여 텍스트를 임베딩합니다.
        오류 발생 시 안전한 임베딩 처리를 제공합니다.
        """
        start_time = time.time()
        thread_id = threading.get_ident()  # 현재 스레드 ID 가져오기
        
        # 모델 로드 확인
        if not self._model_loaded:
            logging.error("Model not properly loaded")
            raise RuntimeError("Embedding model not loaded")
        
        # 입력 텍스트 확인 및 전처리
        if not text:
            logging.warning("Empty text provided for embedding")
            # 빈 텍스트에 대해 0으로 채워진 임베딩 반환
            return [0.0] * 1024
            
        # 텍스트가 튜플인 경우 첫 번째 요소만 사용
        if isinstance(text, tuple):
            logging.warning(f"[Thread-{thread_id}] 텍스트가 튜플 형태로 전달됨, 첫 번째 요소만 사용")
            text = text[0] if len(text) > 0 else ""
            
        # 텍스트가 문자열이 아닌 경우 문자열로 변환
        if not isinstance(text, str):
            logging.warning(f"[Thread-{thread_id}] 텍스트가 문자열이 아님: {type(text)}, 문자열로 변환 시도")
            try:
                text = str(text)
            except Exception as e:
                logging.error(f"[Thread-{thread_id}] 텍스트 변환 실패: {str(e)}")
                return [0.0] * 1024
        
        # 텍스트가 비어있는 경우
        if len(text) == 0:
            logging.warning("Empty text after preprocessing")
            return [0.0] * 1024
        
        try:
            # 임베딩 시작 시간
            embed_start = time.time()
            
            with torch.no_grad():  # 그래디언트 계산 비활성화
                # 텍스트가 너무 길면 잘라내기
                if len(text) > 5000:  # 안전을 위한 최대 길이 제한
                    text = text[:5000]
                    logging.warning(f"[Thread-{thread_id}] 텍스트 길이가 5000자를 초과하여 잘랐습니다.")
                
                # 모드에 따라 적절한 최대 길이 설정
                mode = "gpu" if torch.cuda.is_available() and self.gpu_initialized else "cpu"
                max_length = 512 if mode == "cpu" else 1024  # CPU에서는 더 작은 값 사용
                
                try:
                    # 임베딩 생성
                    embeddings = self.bge_emb.encode(text, max_length=max_length)['dense_vecs']
                except Exception as encode_error:
                    logging.error(f"[Thread-{thread_id}] 임베딩 인코딩 오류: {str(encode_error)}")
                    # 더 간단한 텍스트로 재시도
                    fallback_text = text[:100] + "..." if len(text) > 100 else text
                    try:
                        embeddings = self.bge_emb.encode(fallback_text, max_length=256)['dense_vecs']
                        logging.info(f"[Thread-{thread_id}] 대체 텍스트로 임베딩 성공")
                    except:
                        logging.error(f"[Thread-{thread_id}] 대체 텍스트 임베딩도 실패")
                        return [0.0] * 1024
            
            # 임베딩 소요 시간
            embed_time = time.time() - embed_start
            
            # 임베딩 형태 확인 및 처리
            if isinstance(embeddings, list) and len(embeddings) > 0:
                # 리스트인 경우 첫 번째 요소 사용
                if isinstance(embeddings[0], (list, np.ndarray)):
                    result = embeddings[0]
                elif isinstance(embeddings[0], torch.Tensor):
                    result = embeddings[0].tolist()
                else:
                    result = embeddings[0] if hasattr(embeddings[0], '__iter__') else embeddings
            elif isinstance(embeddings, np.ndarray):
                # numpy 배열인 경우
                if embeddings.ndim > 1 and embeddings.shape[0] > 0:
                    result = embeddings[0].tolist()
                else:
                    result = embeddings.tolist()
            elif isinstance(embeddings, torch.Tensor):
                # tensor인 경우
                if embeddings.dim() > 1 and embeddings.shape[0] > 0:
                    result = embeddings[0].tolist()
                else:
                    result = embeddings.tolist()
            elif isinstance(embeddings, tuple):
                # 튜플인 경우 리스트로 변환
                logging.warning(f"[Thread-{thread_id}] 임베딩 결과가 튜플: {len(embeddings)}")
                result = list(embeddings)
            else:
                logging.warning(f"[Thread-{thread_id}] 예상치 못한 임베딩 형식: {type(embeddings)}")
                result = [0.0] * 1024  # 기본 임베딩 반환
            
            # 결과가 리스트가 아니면 리스트로 변환 시도
            if not isinstance(result, list):
                try:
                    if hasattr(result, 'tolist'):
                        result = result.tolist()
                    elif hasattr(result, '__iter__'):
                        result = list(result)
                    else:
                        logging.warning(f"[Thread-{thread_id}] 임베딩 결과를 리스트로 변환할 수 없음: {type(result)}")
                        result = [0.0] * 1024
                except Exception as convert_error:
                    logging.error(f"[Thread-{thread_id}] 임베딩 결과 변환 오류: {str(convert_error)}")
                    result = [0.0] * 1024
            
            # 결과 차원 확인
            if len(result) != 1024:
                logging.warning(f"[Thread-{thread_id}] 임베딩 벡터 크기가 1024가 아님: {len(result)}")
                # 크기가 맞지 않으면 0으로 채워진 벡터 반환
                result = [0.0] * 1024
            
            # 임베딩 완료 로그 - 핵심 정보만 유지
            end_time = time.time()
            total_time = end_time - start_time
            
            # 비정상적으로 긴 처리 시간 경고 (45초 이상)
            if total_time > 45:
                logging.warning(f"[Thread-{thread_id}] ⚠️ 비정상적으로 긴 임베딩 총 처리 시간: {total_time:.2f}초")
            
            # 성능 요약 로그
            logging.info(f"[Thread-{thread_id}] 임베딩 완료: 총 {total_time:.4f}초, 계산={embed_time:.4f}초, 결과 길이={len(result)}")
            
            return result
            
        except Exception as e:
            logging.error(f"임베딩 생성 오류: {str(e)}")
            import traceback
            logging.error(f"스택 트레이스: {traceback.format_exc()}")
            # 오류 시 기본 임베딩 반환
            return [0.0] * 1024

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

