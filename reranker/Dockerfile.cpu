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
RUN pip install --no-cache-dir -r requirements.txt

# flashrank 패키지 설치 (볼륨 마운트 전 기본 설치)
RUN pip install -e /usr/local/lib/python3.10/site-packages/flashrank

# uwsgi 설치
RUN pip install --no-cache-dir uwsgi

# 프로젝트 파일 복사
COPY . .

# 모델 디렉토리 생성
RUN mkdir -p /reranker/models

# 포트 노출
EXPOSE 9020

# 실행 명령
CMD ["uwsgi", "--ini", "/reranker/uwsgi.ini"] 