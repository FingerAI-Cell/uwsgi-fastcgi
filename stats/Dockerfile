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

# uWSGI 설정
ENV UWSGI_PROCESSES=2
ENV UWSGI_THREADS=4
ENV UWSGI_MASTER=1
ENV UWSGI_SOCKET=/tmp/stats.sock
ENV UWSGI_CHMOD_SOCKET=666
ENV UWSGI_VACUUM=1
ENV UWSGI_DIE_ON_TERM=1
ENV UWSGI_WSGI_FILE=app.py
ENV UWSGI_CALLABLE=app

CMD ["uwsgi", "--ini", "/app/uwsgi.ini"] 