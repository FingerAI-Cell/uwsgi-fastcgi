import json
from pathlib import Path
from tokenizers import AddedToken, Tokenizer
import onnxruntime as ort
import numpy as np
import os
import zipfile
import requests
from tqdm import tqdm
from flashrank.Config import default_model, default_cache_dir, model_url, model_file_map, listwise_rankers, huggingface_rankers, huggingface_model_map
import collections
from typing import Optional, List, Dict, Any
import logging

class RerankRequest:
    """ Represents a reranking request with a query and a list of passages. 
    
    Attributes:
        query (Optional[str]): The query for which the passages need to be reranked.
        passages (List[Dict[str, Any]]): The list of passages to be reranked.
    """

    def __init__(self, query: Optional[str] = None, passages: Optional[List[Dict[str, Any]]] = None):
        self.query: Optional[str] = query
        self.passages: List[Dict[str, Any]] = passages if passages is not None else []

class Ranker:
    """ A ranker class for reranking passages based on a provided query using a pre-trained model.

    Attributes:
        cache_dir (Path): Path to the cache directory where models are stored.
        model_dir (Path): Path to the directory of the specific model being used.
        session (ort.InferenceSession): The ONNX runtime session for making inferences.
        tokenizer (Tokenizer): The tokenizer for text processing.
    """

    def __init__(self, model_name: str = default_model, cache_dir: str = default_cache_dir, max_length: int = 512, log_level: str = "INFO"):
        """ Initializes the Ranker class with specified model and cache settings.

        Args:
            model_name (str): The name of the model to be used.
            cache_dir (str): The directory where models are cached.
            max_length (int): The maximum length of the tokens.
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        
        # Setting up logging
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
        self.logger = logging.getLogger(__name__)

        self.cache_dir: Path = Path(cache_dir)
        self.model_dir: Path = self.cache_dir / model_name
        self.max_length = max_length
        
        # GPU 사용 가능 여부 확인
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {self.device}")
        except ImportError:
            self.device = "cpu"
            self.logger.info("PyTorch not found, using CPU only")
        
        self.llm_model = None
        self.hf_model = None
        self.hf_tokenizer = None
        
        # HuggingFace 모델 사용 시
        if model_name in huggingface_rankers:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                # 모델 이름 가져오기
                hf_model_name = huggingface_model_map[model_name]
                
                self.logger.info(f"Loading HuggingFace model: {hf_model_name}")
                
                # 캐시 디렉토리 설정
                if not self.model_dir.exists():
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                
                # 모델과 토크나이저 로드
                self.hf_tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_name,
                    cache_dir=str(self.model_dir),
                    local_files_only=False
                )
                self.hf_model = AutoModelForSequenceClassification.from_pretrained(
                    hf_model_name,
                    cache_dir=str(self.model_dir),
                    local_files_only=False,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                # GPU로 모델 이동
                if self.device == "cuda":
                    self.hf_model.to(self.device)
                self.hf_model.eval()
            except ImportError:
                raise ImportError("Please install torch and transformers to use HuggingFace models: pip install torch transformers")
            except Exception as e:
                self.logger.error(f"Failed to load HuggingFace model {hf_model_name}: {str(e)}")
                raise
        # 기존 모델 사용 시
        else:
            self._prepare_model_dir(model_name)
            model_file = model_file_map[model_name]
            
            if model_name in listwise_rankers:
                try:
                    from llama_cpp import Llama
                    # GPU 지원을 위한 옵션 설정
                    gpu_layers = -1 if self.device == "cuda" else 0
                    self.llm_model = Llama(
                        model_path=str(self.model_dir / model_file),
                        n_ctx=max_length,
                        n_threads=8,
                        n_gpu_layers=gpu_layers  # GPU 사용 시 모든 레이어를 GPU로
                    )
                    self.logger.info(f"LLM model loaded with GPU layers: {gpu_layers}")
                except ImportError:
                    raise ImportError("Please install llama-cpp-python with GPU support: CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python")
            else:
                # ONNX Runtime providers 설정
                providers = []
                if self.device == "cuda":
                    providers.extend([
                        ('CUDAExecutionProvider', {
                            'device_id': 0,
                            'arena_extend_strategy': 'kNextPowerOfTwo',
                            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                            'do_copy_in_default_stream': True,
                        }),
                        'CPUExecutionProvider'
                    ])
                else:
                    providers.append('CPUExecutionProvider')
                    
                self.logger.info(f"Using ONNX Runtime providers: {providers}")
                self.session = ort.InferenceSession(
                    str(self.model_dir / model_file),
                    providers=providers
                )
                self.tokenizer: Tokenizer = self._get_tokenizer(max_length)

    def _prepare_model_dir(self, model_name: str):
        """ Ensures the model directory is prepared by downloading and extracting the model if not present.

        Args:
            model_name (str): The name of the model to be prepared.
        """
        # HuggingFace 모델인 경우 다운로드 로직을 건너뜁니다
        if model_name in huggingface_rankers:
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            if not self.model_dir.exists():
                self.model_dir.mkdir(parents=True, exist_ok=True)
            return

        # 일반 모델 처리
        if not self.cache_dir.exists():
            self.logger.debug(f"Cache directory {self.cache_dir} not found. Creating it..")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.model_dir.exists():
            if model_name in huggingface_rankers and use_direct_hf_download:
                self.logger.info(f"Model directory will be created by HuggingFace Hub...")
                self.model_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.logger.info(f"Downloading {model_name}...")
                self._download_model_files(model_name)

    def _download_model_files(self, model_name: str):
        """ Downloads and extracts the model files from a specified URL.

        Args:
            model_name (str): The name of the model to download.
        """
        local_zip_file = self.cache_dir / f"{model_name}.zip"
        formatted_model_url = model_url.format(model_name)
        
        with requests.get(formatted_model_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_zip_file, 'wb') as f, tqdm(desc=local_zip_file.name, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)
        os.remove(local_zip_file)

    def _get_tokenizer(self, max_length: int = 512) -> Tokenizer:
        """ Initializes and configures the tokenizer with padding and truncation.

        Args:
            max_length (int): The maximum token length for truncation.

        Returns:
            Tokenizer: Configured tokenizer for text processing.
        """
        with open(str(self.model_dir / "config.json")) as config_file:
            config = json.load(config_file)
        with open(str(self.model_dir / "tokenizer_config.json")) as tokenizer_config_file:
            tokenizer_config = json.load(tokenizer_config_file)
        with open(str(self.model_dir / "special_tokens_map.json")) as tokens_map_file:
            tokens_map = json.load(tokens_map_file)
        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))

        tokenizer.enable_truncation(max_length=min(tokenizer_config["model_max_length"], max_length))
        tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token=tokenizer_config["pad_token"])

        for token in tokens_map.values():
            if isinstance(token, str):
                tokenizer.add_special_tokens([token])
            elif isinstance(token, dict):
                tokenizer.add_special_tokens([AddedToken(**token)])

        vocab_file = self.model_dir / "vocab.txt"
        if vocab_file.exists():
            tokenizer.vocab = self._load_vocab(vocab_file)
            tokenizer.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in tokenizer.vocab.items()])
        return tokenizer

    def _load_vocab(self, vocab_file: Path) -> Dict[str, int]:
        """ Loads the vocabulary from a file and returns it as an ordered dictionary.

        Args:
            vocab_file (Path): The file path to the vocabulary.

        Returns:
            Dict[str, int]: An ordered dictionary mapping tokens to their respective indices.
        """
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab
    
    def _get_prefix_prompt(self, query, num):
        return [
            {
                "role": "system",
                "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
            },
            {
                "role": "user",
                "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]

    def _get_postfix_prompt(self, query, num):
        example_ordering = "[2] > [1]"
        return {
            "role": "user",
            "content": f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain.",
        }

    def rerank(self, request: RerankRequest) -> List[Dict[str, Any]]:
        """ Reranks a list of passages based on a query using a pre-trained model.

        Args:
            request (RerankRequest): The request containing the query and passages to rerank.

        Returns:
            List[Dict[str, Any]]: The reranked list of passages with added scores.
        """
        query = request.query
        passages = request.passages

        # HuggingFace 모델 사용 (한국어 reranker)
        if self.hf_model is not None:
            self.logger.debug("Running HuggingFace reranking...")
            import torch
            
            pairs = [[query, passage["text"]] for passage in passages]
            
            with torch.no_grad():
                inputs = self.hf_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
                # GPU로 입력 데이터 이동
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.hf_model(**inputs, return_dict=True).logits.view(-1, ).float()
                # CPU로 결과 이동 후 numpy 변환
                if self.device == "cuda":
                    scores = scores.cpu()
                scores = scores.numpy()
            
            # 점수를 sigmoid 함수로 0~1 사이로 정규화
            scores = 1 / (1 + np.exp(-scores))
            
            # 점수 할당
            for score, passage in zip(scores, passages):
                passage["score"] = float(score)
            
            # 점수 기준으로 내림차순 정렬
            passages.sort(key=lambda x: x["score"], reverse=True)
            return passages

        # LLM 방식 (Listwise ranking)
        elif self.llm_model is not None:
            self.logger.debug("Running listwise ranking..")
            num_of_passages = len(passages)
            messages = self._get_prefix_prompt(query, num_of_passages)

            result_map = {}
            for rank, passage in enumerate(passages):
                messages.append(
                    {
                        "role": "user",
                        "content": f"[{rank + 1}] {passage['text']}",
                    }
                )
                messages.append(
                        {
                            "role": "assistant", 
                            "content": f"Received passage [{rank + 1}]."
                        }
                )
                
                result_map[rank + 1] = passage

            messages.append(self._get_postfix_prompt(query, num_of_passages))
            raw_ranks = self.llm_model.create_chat_completion(messages)
            results = []
            for rank in raw_ranks["choices"][0]["message"]["content"].split(" > "):
                results.append(result_map[int(rank.strip("[]"))])
            return results    

        # ONNX 모델 방식 (Pairwise ranking)
        else:
            self.logger.debug("Running pairwise ranking..")
            query_passage_pairs = [[query, passage["text"]] for passage in passages]

            input_text = self.tokenizer.encode_batch(query_passage_pairs)
            input_ids = np.array([e.ids for e in input_text])
            token_type_ids = np.array([e.type_ids for e in input_text])
            attention_mask = np.array([e.attention_mask for e in input_text])

            use_token_type_ids = token_type_ids is not None and not np.all(token_type_ids == 0)

            onnx_input = {
                "input_ids": input_ids.astype(np.int64), 
                "attention_mask": attention_mask.astype(np.int64)
            }
            if use_token_type_ids:
                onnx_input["token_type_ids"] = token_type_ids.astype(np.int64)

            outputs = self.session.run(None, onnx_input)
            logits = outputs[0]

            if logits.shape[1] == 1:
                scores = 1 / (1 + np.exp(-logits.flatten()))
            else:
                exp_logits = np.exp(logits)
                scores = exp_logits[:, 1] / np.sum(exp_logits, axis=1)

            for score, passage in zip(scores, passages):
                passage["score"] = score

            passages.sort(key=lambda x: x["score"], reverse=True)
            return passages
