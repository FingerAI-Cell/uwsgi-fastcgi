FROM ollama/ollama:latest

# 기본 도구 설치 (여러 리눅스 배포판 지원)
RUN if command -v apt-get >/dev/null; then \
      apt-get update && apt-get install -y bash; \
    elif command -v yum >/dev/null; then \
      yum -y install bash; \
    elif command -v apk >/dev/null; then \
      apk add --no-cache bash; \
    fi

# 작업 디렉토리 설정
WORKDIR /app

# 초기화 스크립트 복사
COPY init.sh /app/
COPY models.txt /app/
RUN chmod +x /app/init.sh

# Ollama 실행을 위한 기본 포트 노출
EXPOSE 11434

# 실행 명령 (단순히 serve만 실행)
CMD ["serve"] 