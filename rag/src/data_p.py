import pandas as pd 
import numpy as np 
import os 

class DataProcessor():
    def __init__(self, args):
        self.args = args 
    
    def cleanse_text(self, text):
        '''
        다중 줄바꿈 제거 및 특수 문자 중복 제거
        '''
        import re 
        text = re.sub(r'(\n\s*)+\n+', '\n\n', text)
        text = re.sub(r"\·{1,}", " ", text)
        text = re.sub(r"\.{1,}", ".", text)
        return text

    def check_l2_threshold(self, txt, threshold, value):
        threshold_txt = '' 
        print(f'Euclidean Distance: {value}, Threshold: {threshold}')
        if value > threshold:
            threshold_txt = '모르는 정보입니다.'
        else:
            threshold_txt = txt 
        return threshold_txt

    def cohere_rerank(self, data):
        pass

class VectorProcessor:
    def set_gpu(self, model):
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        model.to(self.device)
    
    def set_emb_model(self, model_type):
        if model_type == 'bge':
            from FlagEmbedding import BGEM3FlagModel
            model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
            return model
        
    def set_embbeding_config(self, batch_size=12, max_length=1024):
        self.emb_config = {
            "batch_size": batch_size, 
            "max_length": max_length 
        }
    
    def embed_data(self, model, text):
        if isinstance(text, str):
            # encode result  => dense_vecs, lexical weights, colbert_vecs
            embeddings = model.encode(text, batch_size=self.emb_config['batch_size'], max_length=self.emb_config['max_length'])['dense_vecs']
        else:       
            embeddings = model.encode(list(text), batch_size=self.emb_config['batch_size'], max_length=self.emb_config['max_length'])['dense_vecs']  
        embeddings = list(map(np.float32, embeddings))
        return embeddings    

    def calc_emb_similarity(self, emb1, emb2, metric='L2'):
        if metric == 'L2':   # Euclidean distance
            l2_distance = np.linalg.norm(emb1 - emb2)
            return l2_distance