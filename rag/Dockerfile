# 기본 Docker Image 설정   
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel    
LABEL description='Docker image created for rag architecture'

# Docker 내 작업 경로 설정 
WORKDIR /rag 
COPY . .

# 환경 변수 설정
ENV LC_ALL ko_KR.UTF-8 
# PyTorch 컴파일러 관련 오류 해결을 위한 설정
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_CACHE=/rag/models
ENV HF_HOME=/rag/models
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# 패키지 설치 및 캐시 정리 (한 번의 RUN 명령으로 레이어 수 감소)
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales build-essential curl && \
    locale-gen ko_KR.UTF-8 && \
    apt-get install -y --no-install-recommends python3-pip && \
    #pip install --no-cache-dir uwsgi && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install numpy==1.24.4

# 디버깅용 명령
RUN ls -la /rag/
RUN python -c "import sys; print(sys.path)"

# FastCGI 설정으로 실행
CMD ["uwsgi", "--ini", "/rag/uwsgi.ini"]