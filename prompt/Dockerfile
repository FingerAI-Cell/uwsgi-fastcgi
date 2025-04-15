FROM python:3.10-slim

# 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /prompt

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# uwsgi 설치 - pip를 통해 설치하여 Python 3.10과 호환되도록 함
RUN pip install uwsgi

# 로그 디렉토리 생성
RUN mkdir -p /var/log/prompt

# 프로젝트 파일 복사
COPY . .

# Flask가 설치되었는지 확인 (디버깅 용도)
RUN python -c "import flask; print('Flask 설치됨')"

# 포트 노출 (필요시)
EXPOSE 5000

# 직접 uwsgi 명령어 실행 (ini 파일 대신 직접 옵션 지정)
CMD ["uwsgi", "--socket", "/tmp/prompt.sock", "--protocol", "fastcgi", "--module", "app:app", "--master", "--processes", "4", "--threads", "2", "--chmod-socket=666", "--logto", "/var/log/prompt/uwsgi.log"] 