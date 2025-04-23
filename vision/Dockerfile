FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /vision

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 필요한 디렉토리 생성
RUN mkdir -p /vision/uploads /var/log/vision

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 권한 설정
RUN chmod -R 755 /vision \
    && chmod -R 777 /vision/uploads \
    && chmod -R 777 /var/log/vision

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# uWSGI로 실행 (Prompt 서비스와 동일한 방식)
CMD ["uwsgi", "--socket", "/tmp/vision.sock", "--protocol", "fastcgi", "--module", "app:app", "--master", "--processes", "4", "--threads", "2", "--chmod-socket=666", "--logto", "/var/log/vision/uwsgi.log"] 