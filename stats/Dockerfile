FROM python:3.10-slim

WORKDIR /app

# 빌드 도구 및 컴파일러 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 로그 디렉토리 생성 및 권한 설정
RUN mkdir -p /var/log/nginx && \
    touch /var/log/nginx/api-stats.log && \
    chmod 666 /var/log/nginx/api-stats.log

# 로그 파일 경로 설정
ENV NGINX_LOG_PATH=/var/log/nginx/api-stats.log
ENV LOG_CHECK_INTERVAL=5

# uWSGI 설정
ENV UWSGI_PROCESSES=3
ENV UWSGI_THREADS=4
ENV UWSGI_MASTER=1
ENV UWSGI_PROTOCOL=fastcgi
ENV UWSGI_SOCKET=/tmp/stats.sock
ENV UWSGI_CHMOD_SOCKET=666
ENV UWSGI_VACUUM=1
ENV UWSGI_DIE_ON_TERM=1
ENV UWSGI_WSGI_FILE=app.py
ENV UWSGI_CALLABLE=app

# 볼륨 설정
VOLUME ["/var/log/nginx"]

CMD ["uwsgi", "--ini", "/app/uwsgi.ini"] 