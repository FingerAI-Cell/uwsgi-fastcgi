FROM python:3.10-slim

WORKDIR /api-gateway

# 필요한 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# uwsgi 설정을 위한 디렉토리 생성
RUN mkdir -p /tmp

# 포트 노출
EXPOSE 3000

# 기본 명령어
CMD ["uwsgi", "--ini", "uwsgi.ini"] 