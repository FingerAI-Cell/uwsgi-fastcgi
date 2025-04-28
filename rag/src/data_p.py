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

    def chunk_text(self, text, max_bytes=512, overlap_bytes=256):
        """
        벡터 임베딩 전, 텍스트를 바이트 단위로 분할.
        - max_bytes: 청크당 최대 바이트 수 (기본값 512바이트)
        - overlap_bytes: 청크 간 중복되는 바이트 수 (기본값 256바이트)
        - (text_chunk, chunk_no) 리스트 반환
        """
        if not text:
            print("[WARNING] Empty or None text input")
            return [(text or "", 1)]
            
        if max_bytes <= 0:
            print("[WARNING] Invalid max_bytes value, using default 512")
            max_bytes = 512
            
        if overlap_bytes >= max_bytes:
            print("[WARNING] overlap_bytes too large, adjusting to max_bytes/2")
            overlap_bytes = max_bytes // 2

        text_bytes = text.encode('utf-8')
        total_bytes = len(text_bytes)
        print(f"[DEBUG] Input text: {len(text)} chars, {total_bytes} bytes")
        
        if total_bytes <= max_bytes:
            print("[DEBUG] Text bytes within max_bytes, returning single chunk")
            return [(text, 1)]
        
        chunks = []
        chunk_no = 1
        start_idx = 0
        
        while start_idx < len(text):
            # 현재 위치에서 시작하는 가능한 최대 길이의 청크 찾기
            end_idx = start_idx
            current_chunk = text[start_idx:end_idx+1].encode('utf-8')
            
            while end_idx < len(text) and len(current_chunk) < max_bytes:
                end_idx += 1
                if end_idx < len(text):
                    current_chunk = text[start_idx:end_idx+1].encode('utf-8')
            
            # 마지막 문자가 max_bytes를 초과하면 한 문자 뒤로
            if len(current_chunk) > max_bytes and end_idx > start_idx:
                end_idx -= 1
                current_chunk = text[start_idx:end_idx].encode('utf-8')
            
            chunk = text[start_idx:end_idx]
            chunk_bytes = len(chunk.encode('utf-8'))
            print(f"[DEBUG] Creating chunk {chunk_no}: pos {start_idx}-{end_idx} (bytes={chunk_bytes})")
            
            if chunk:  # 빈 청크는 추가하지 않음    
                chunks.append((chunk, chunk_no))
                chunk_no += 1
            
            if end_idx >= len(text):
                break
                
            # 다음 시작 위치 계산 (overlap 고려)
            next_start = end_idx - (overlap_bytes // 2)  # overlap_bytes의 절반만큼 뒤로 이동
            if next_start <= start_idx:  # 진전이 없으면
                next_start = start_idx + 1  # 최소 1문자는 전진
            
            start_idx = next_start
        
        print(f"[DEBUG] Created {len(chunks)} chunks")
        for i, (chunk, no) in enumerate(chunks):
            bytes_len = len(chunk.encode('utf-8'))
            print(f"[DEBUG] Chunk {i+1}: bytes={bytes_len}, id={no}")
            if bytes_len > max_bytes:
                print(f"[WARNING] Chunk {i+1} exceeds max_bytes: {bytes_len} > {max_bytes}")
        
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
