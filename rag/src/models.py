from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import TextStreamer, GenerationConfig
from openai import OpenAI
from abc import ABC, abstractmethod
import numpy as np
import torch
import warnings
import torch
import os
import logging

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
            logging.info(f"GPU is available. Using device: {self.device}")
            logging.info(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
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
            self.bge_emb = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=self.device)
            logging.info(f"Loaded BGE model on device: {self.device}")
            if torch.cuda.is_available():
                logging.info(f"GPU Memory after model load: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        
             
    def bge_embed_data(self, text):
        if torch.cuda.is_available():
            logging.info(f"GPU Memory before embedding: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
        batch_size = self.emb_config['batch_size']
        logging.info(f"Using batch size: {batch_size}")
        
        if isinstance(text, str):
            # encode result  => dense_vecs, lexical weights, colbert_vecs
            embeddings = self.bge_emb.encode(text, batch_size=batch_size, max_length=self.emb_config['max_length'])['dense_vecs']
            logging.info(f"Embedded single text with batch size {batch_size}")
        else:       
            total_samples = len(text)
            logging.info(f"Embedding {total_samples} texts with batch size {batch_size}")
            embeddings = self.bge_emb.encode(list(text), batch_size=batch_size, max_length=self.emb_config['max_length'])['dense_vecs']  
            logging.info(f"Completed embedding {total_samples} texts in {total_samples//batch_size + (1 if total_samples%batch_size else 0)} batches")
            
        if torch.cuda.is_available():
            logging.info(f"GPU Memory after embedding: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
        embeddings = list(map(np.float32, embeddings))
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

