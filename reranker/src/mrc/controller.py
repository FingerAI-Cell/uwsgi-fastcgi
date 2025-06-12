import json
import torch
import torch.nn.functional as F
import logging

from munch import munchify
from typing import List, Dict, Any

from .ifv_module import IFVModule, get_model_config, get_model

# 로깅 설정
logger = logging.getLogger(__name__)

class MRCController:
    """MRC 모델을 관리하고 추론 기능을 제공하는 컨트롤러"""
    
    _instance = None  # 싱글톤 인스턴스
    
    @classmethod
    def get_instance(cls, config_path=None, model_path=None):
        """싱글톤 패턴으로 인스턴스 반환"""
        if cls._instance is None:
            logger.info(f"Creating new MRCController instance with config: {config_path}")
            cls._instance = cls(config_path, model_path)
        else:
            logger.info("Returning existing MRCController instance")
        return cls._instance
    
    def __init__(self, config_path, model_path):
        """
        MRC 컨트롤러 초기화
        
        Args:
            config_path: 모델 설정 경로
            model_path: 모델 체크포인트 경로
        """
        self.config_path = config_path
        self.model_path = model_path
        
        # 모델 로드
        logger.info("Loading MRC model...")
        self.model_config_args = get_model_config(self.config_path, plm='koroberta', infer_batch_size=10)
        self.model = get_model(self.model_config_args, self.model_path)
        logger.info("MRC model loaded successfully")
        
        # 소프트맥스 함수 정의
        self.softmax = lambda logits: F.softmax(logits.to(dtype=torch.float32), dim=-1)
    
    def infer_single(self, question: str, context: str, temperature: float = 1.0) -> Dict[str, Any]:
        """
        단일 질문-문맥 쌍에 대해 MRC 추론 수행
        
        Args:
            question: 질문
            context: 문맥(지문)
            temperature: 답변 확률 조정 온도 계수
            
        Returns:
            MRC 모델의 예측 결과 (정답, 위치, 확률)
        """
        tokenized = self.model.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[(question, context)],
            add_special_tokens=True,
            padding=False,
            truncation='only_second',
            max_length=self.model_config_args.max_seq_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_tensors='pt')
            
        results = self._infer_batch(tokenized, [context], [temperature])
        return results[0]
    
    def infer_multi(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        여러 질문-문맥 쌍에 대해 MRC 추론 수행
        
        Args:
            samples: 질문-문맥 쌍의 리스트
            
        Returns:
            MRC 모델의 예측 결과 리스트
        """
        # 질문-문맥 쌍과 온도 추출
        question_context_pairs = [(s['question'], s['context']) for s in samples]
        contexts = [s['context'] for s in samples]
        temperatures = [s.get('temperature', 1.0) for s in samples]
        
        # 토크나이징
        tokenized = self.model.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=question_context_pairs,
            add_special_tokens=True,
            padding=True,
            truncation='only_second',
            max_length=self.model_config_args.max_seq_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_tensors='pt')
            
        # 배치 처리
        results = []
        for i_batch in range(((len(samples) - 1) // self.model_config_args.infer_batch_size) + 1):
            s_idx = self.model_config_args.infer_batch_size * i_batch
            t_idx = self.model_config_args.infer_batch_size * (i_batch + 1)
            batch_contexts = contexts[s_idx:t_idx]
            batch_temperatures = temperatures[s_idx:t_idx]
            batch_results = self._infer_batch(tokenized, batch_contexts, batch_temperatures, s_idx, t_idx)
            results.extend(batch_results)
            
        return results
    
    def _infer_batch(self, tokenized, contexts, temperatures, s_idx=0, t_idx=None):
        """내부 배치 추론 메소드"""
        results = []
        
        with torch.no_grad():
            device = self.model_config_args.device
            output = self.model(
                input_ids=tokenized['input_ids'][s_idx:t_idx].to(device),
                attention_mask=tokenized['attention_mask'][s_idx:t_idx].to(device),
                token_type_ids=tokenized['token_type_ids'][s_idx:t_idx].to(device))
            output = {k: logits.detach().cpu() for k, logits in output.items()}
            
            # 답변 가능성 점수 계산
            ans_scores = self.softmax(output['ans_logits'])[:,1].tolist()
            if temperatures:
                ans_scores = [score / temp for score, temp in zip(ans_scores, temperatures)]
                
            # 시작/끝 위치 예측
            start_logits, end_logits = output['start_logits'], output['end_logits']
            s = self.softmax(start_logits).unsqueeze(-1)  # (bat, len, 1)
            t = self.softmax(end_logits).unsqueeze(-2)    # (bat, 1, len)
            p = torch.triu(s + t, diagonal=1) / 2  # 상삼각 행렬 (bat, len, len)
            st_list = [(p[i] == torch.max(p[i])).nonzero().tolist()[0] for i in range(p.size(0))]
            
            # 결과 변환
            for i_sample, (s_tok_id, t_tok_id) in enumerate(st_list):
                s_char_id = int(tokenized['offset_mapping'][s_idx:t_idx][i_sample][s_tok_id][0])
                t_char_id = int(tokenized['offset_mapping'][s_idx:t_idx][i_sample][t_tok_id-1][1])
                pred_str = contexts[i_sample][s_char_id:t_char_id]
                
                results.append({
                    'answer': pred_str,
                    'char_ids': [s_char_id, t_char_id],
                    'answerability': ans_scores[i_sample]
                })
                
        return results 