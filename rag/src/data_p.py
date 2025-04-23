import pandas as pd 
import numpy as np 
import hashlib
import os 

class DataProcessor:    
    def cleanse_text(self, text):
        '''
        다중 줄바꿈 제거 및 특수 문자 중복 제거
        '''
        import re 
        text = re.sub(r'(\n\s*)+\n+', '\n\n', text)
        text = re.sub(r"\·{1,}", " ", text)
        text = re.sub(r"\.{1,}", ".", text)
        return text

    def get_text_bytes(self, text):
        """
        텍스트의 바이트 길이를 반환하고, 주어진 최대 바이트 수에 맞는 실제 문자 인덱스를 찾습니다.
        """
        encoded = text.encode('utf-8')
        return len(encoded)

    def find_byte_boundary(self, text, max_bytes):
        """
        주어진 최대 바이트 수에 맞는 실제 문자 인덱스를 찾습니다.
        """
        encoded = text.encode('utf-8')
        if len(encoded) <= max_bytes:
            return len(text)
        
        # 바이너리 서치로 적절한 문자 위치 찾기
        left, right = 0, len(text)
        while left < right:
            mid = (left + right) // 2
            if len(text[:mid].encode('utf-8')) <= max_bytes:
                left = mid + 1
            else:
                right = mid
        
        # 정확한 바이트 경계를 찾았으면 그 이전 문자까지 반환
        return left - 1

    def chunk_text(self, text, max_bytes=500, overlap_bytes=250):
        """
        벡터 임베딩 전, 텍스트를 바이트 단위로 분할.
        - max_bytes: 청크당 최대 바이트 수 (기본값 500바이트)
        - overlap_bytes: 청크 간 중복되는 바이트 수 (기본값 250바이트)
        - (text_chunk, chunk_no) 리스트 반환
        """
        if self.get_text_bytes(text) <= max_bytes:
            return [(text, 1)]
        
        chunks = []
        chunk_no = 1
        current_pos = 0
        
        while current_pos < len(text):
            # 현재 위치에서 시작하는 텍스트에서 max_bytes에 맞는 문자 위치 찾기
            if current_pos == 0:
                # 첫 번째 청크는 overlap 없이 max_bytes까지
                end_pos = self.find_byte_boundary(text[current_pos:], max_bytes)
                chunk = text[current_pos:current_pos + end_pos]
                chunks.append((chunk, chunk_no))
                current_pos = current_pos + end_pos - self.find_byte_boundary(text[current_pos:current_pos + end_pos], overlap_bytes)
            else:
                # 이후 청크는 이전 청크와 overlap_bytes만큼 중복
                end_pos = self.find_byte_boundary(text[current_pos:], max_bytes)
                if end_pos == 0:  # 남은 텍스트가 없으면 종료
                    break
                chunk = text[current_pos:current_pos + end_pos]
                chunks.append((chunk, chunk_no))
                # 다음 시작 위치 계산 (현재 청크의 끝에서 overlap_bytes만큼 앞으로)
                current_pos = current_pos + end_pos - self.find_byte_boundary(text[current_pos:current_pos + end_pos], overlap_bytes)
            
            chunk_no += 1
            
            # 마지막 청크가 너무 작으면 이전 청크와 병합
            if chunk_no > 1 and current_pos < len(text) and self.get_text_bytes(text[current_pos:]) < overlap_bytes:
                last_chunk, last_no = chunks.pop()
                merged_chunk = text[current_pos - self.find_byte_boundary(text[current_pos - len(last_chunk):current_pos], max_bytes):]
                chunks.append((merged_chunk, last_no))
                break
        
        return chunks

    def check_l2_threshold(self, txt, threshold, value):
        threshold_txt = '' 
        print(f'Euclidean Distance: {value}, Threshold: {threshold}')
        if value > threshold:
            threshold_txt = '모르는 정보입니다.'
        else:
            threshold_txt = txt 
        return threshold_txt

    def hash_text(self, text, hash_type):
        if hash_type == 'blake':
            hashed_text = hashlib.blake2b(text.encode()).hexdigest() 
        elif hash_type == 'sha256':
            hashed_text = hashlib.sha256(text.encode()).hexdigest()
        return hashed_text

    def cohere_rerank(self, data):
        pass
