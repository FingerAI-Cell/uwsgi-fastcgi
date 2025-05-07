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
        벡터 임베딩 전, 텍스트를 바이트 단위로 분할하되 단어 단위를 보존.
        - max_bytes: 청크당 최대 바이트 수 (기본값 512바이트)
        - overlap_bytes: 청크 간 중복되는 바이트 수 (기본값 256바이트)
        - (text_chunk, chunk_no) 리스트 반환
        
        새로운 알고리즘:
        1. 기본 단위는 half_max_bytes(256바이트)로 설정
        2. 첫 청크는 두 개의 half_max_bytes 단위로 구성 (총 max_bytes)
        3. 이후 청크들은 이전 청크의 뒷부분 half_max_bytes + 새로운 half_max_bytes로 구성
        4. half_max_bytes를 자를 때 가능하면 공백 위치를 고려
        """
        if not text:
            print("[WARNING] Empty or None text input")
            return [(text or "", 1)]
            
        if max_bytes <= 0:
            print("[WARNING] Invalid max_bytes value, using default 512")
            max_bytes = 512
            
        half_max_bytes = max_bytes // 2  # 기본 단위 (256바이트)
        search_margin = 50  # 공백 찾기 위한 여유 범위 (바이트)
        
        print(f"[INFO] 청킹 알고리즘 설정: max_bytes={max_bytes}, half_max_bytes={half_max_bytes}, search_margin={search_margin}")
        
        text_bytes = text.encode('utf-8')
        total_bytes = len(text_bytes)
        print(f"[DEBUG] Input text: {len(text)} chars, {total_bytes} bytes")
        
        if total_bytes <= max_bytes:
            print("[DEBUG] Text bytes within max_bytes, returning single chunk")
            return [(text, 1)]
        
        chunks = []
        chunk_no = 1
        pos = 0  # 현재 위치 (바이트 단위)
        
        # 첫 번째 블록 자르기
        first_block_end = self._find_optimal_boundary(text, pos, half_max_bytes, search_margin)
        print(f"[DEBUG] 첫 번째 블록 경계 계산: {pos} -> {first_block_end} ({first_block_end-pos} 글자)")
        
        # 텍스트 길이 전체를 처리할 때까지 반복
        while pos < len(text):
            print(f"[DEBUG] 청킹 진행 중: 현재 위치={pos}/{len(text)} ({(pos/len(text)*100):.1f}%)")
            
            # 첫 번째 블록 경계 계산
            block1_end = self._find_optimal_boundary(text, pos, half_max_bytes, search_margin)
            block1_bytes = len(text[pos:block1_end].encode('utf-8'))
            print(f"[DEBUG] 첫 번째 블록: pos {pos}-{block1_end} ({block1_end-pos} 글자, {block1_bytes} 바이트)")
            
            # 두 번째 블록 경계 계산
            block2_end = self._find_optimal_boundary(text, block1_end, half_max_bytes, search_margin)
            block2_bytes = len(text[block1_end:block2_end].encode('utf-8'))
            print(f"[DEBUG] 두 번째 블록: pos {block1_end}-{block2_end} ({block2_end-block1_end} 글자, {block2_bytes} 바이트)")
            
            # 두 블록을 합쳐서 하나의 청크 생성
            chunk = text[pos:block2_end].strip()
            chunk_bytes = len(chunk.encode('utf-8'))
            print(f"[DEBUG] Creating chunk {chunk_no}: pos {pos}-{block2_end} (글자수={len(chunk)}, bytes={chunk_bytes})")
            
            if chunk:  # 빈 청크는 추가하지 않음
                chunks.append((chunk, chunk_no))
                chunk_no += 1
            
            if block2_end >= len(text):
                print(f"[DEBUG] 텍스트 끝에 도달: {block2_end}/{len(text)}")
                break
                
            # 다음 청크의 시작점은 첫 번째 블록의 끝점
            # (이렇게 하면 두 번째 블록이 중복 영역이 됨)
            pos = block1_end
            print(f"[DEBUG] 다음 청크 시작점 설정: pos={pos}")
        
        print(f"[INFO] 청킹 완료: 총 {len(chunks)}개 청크 생성")
        for i, (chunk, no) in enumerate(chunks):
            bytes_len = len(chunk.encode('utf-8'))
            print(f"[DEBUG] Chunk {i+1}: id={no}, 글자수={len(chunk)}, bytes={bytes_len}")
        
        return chunks

    def _find_optimal_boundary(self, text, start_pos, target_bytes, margin):
        """
        최적의 청크 경계를 찾는 헬퍼 함수
        
        Args:
            text: 원본 텍스트
            start_pos: 시작 위치 (문자 인덱스)
            target_bytes: 목표 바이트 크기 (일반적으로 256)
            margin: 공백 찾기 위한 여유 범위 (바이트)
            
        Returns:
            최적의 경계 위치 (문자 인덱스)
        """
        if start_pos >= len(text):
            print(f"[DEBUG] 시작 위치({start_pos})가 텍스트 길이({len(text)})보다 크거나 같음")
            return len(text)
        
        # 기본 경계값 계산 (정확히 target_bytes)
        end_pos = start_pos
        min_bytes = max(1, target_bytes - margin)  # 최소 바이트 (여유 범위 고려)
        
        # target_bytes 바이트에 해당하는 문자 위치 찾기
        current_bytes = 0
        while end_pos < len(text) and current_bytes < target_bytes:
            current_bytes = len(text[start_pos:end_pos+1].encode('utf-8'))
            if current_bytes <= target_bytes:
                end_pos += 1
            else:
                break
        
        # 안전 검사: end_pos가 start_pos보다 최소 1 이상 커야 함
        end_pos = max(start_pos + 1, end_pos)
        print(f"[DEBUG] 기본 경계 계산: start_pos={start_pos}, end_pos={end_pos}, bytes={current_bytes}")
        
        # 최적 경계 위치 (공백 기준)
        optimal_pos = end_pos - 1  # 일단 경계 직전 위치로 설정
        
        # 목표 크기에 근접하면서 공백을 찾음
        # 단, min_bytes 이상은 되어야 함
        backward_pos = end_pos - 1
        found_space = False
        
        while backward_pos > start_pos:
            current_bytes = len(text[start_pos:backward_pos+1].encode('utf-8'))
            if current_bytes < min_bytes:
                print(f"[DEBUG] 최소 바이트({min_bytes})보다 작아져서 공백 탐색 중단: 현재={current_bytes}바이트")
                break  # 최소 바이트 크기보다 작아지면 중단
            
            if text[backward_pos].isspace():
                optimal_pos = backward_pos + 1  # 공백 다음 위치
                found_space = True
                print(f"[DEBUG] 공백 발견: 위치={backward_pos}, 최적 위치={optimal_pos}")
                break
            
            backward_pos -= 1
        
        if not found_space:
            print(f"[DEBUG] 공백을 찾지 못했음: 기본 경계 사용={optimal_pos}")
        
        # 공백을 찾지 못하면 원래 계산된 경계 사용
        result = min(len(text), optimal_pos)
        print(f"[DEBUG] 최종 경계 위치: {result} (텍스트 길이={len(text)})")
        return result

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
