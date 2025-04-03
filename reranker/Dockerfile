FROM python:3.10-slim

WORKDIR /reranker

# 기본 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flashrank

# uwsgi 설치
RUN pip install --no-cache-dir uwsgi

# 프로젝트 파일 복사
COPY . .

# 모델 디렉토리 생성
RUN mkdir -p /reranker/models

# 모델 다운로드는 런타임에 환경변수를 통해 지정된 모델을 사용하도록 주석 처리
# 필요한 경우 여기에서 기본 모델을 다운로드할 수 있습니다
# RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-TinyBERT-L-2-v2', cache_dir='/reranker/models')"

# 포트 노출
EXPOSE 9020

# 실행 명령
CMD ["uwsgi", "--ini", "/reranker/uwsgi.ini"] 