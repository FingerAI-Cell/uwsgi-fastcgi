#!/bin/bash

echo "=== 서비스 안전 종료 시작 ==="

# 현재 디렉토리 확인 및 루트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR" || exit 1

# sudo가 필요한지 확인
DOCKER_CMD="docker"
if ! $DOCKER_CMD ps > /dev/null 2>&1; then
    if sudo -n true 2>/dev/null; then
        DOCKER_CMD="sudo docker"
        echo "sudo 권한으로 Docker 명령을 실행합니다."
    else
        echo "Warning: Docker는 sudo 권한이 필요할 수 있습니다. 실행 중 오류가 발생하면 sudo를 사용하세요."
    fi
fi

# 환경 변수 로드
set -a
source .env
set +a

# 서비스 종료
case "$1" in
  "all")
    echo "🛑 모든 서비스 종료 중... (CPU 모드)"
    $DOCKER_CMD compose --profile all --profile cpu-only down
    ;;
  "all-gpu")
    echo "🛑 모든 서비스 종료 중... (GPU 모드)"
    $DOCKER_CMD compose --profile all --profile gpu-only down
    ;;
  "rag")
    echo "🛑 RAG 서비스 종료 중..."
    $DOCKER_CMD compose --profile rag-only down
    ;;
  "reranker")
    echo "🛑 Reranker 서비스 종료 중..."
    $DOCKER_CMD compose --profile reranker-only down
    ;;
  "prompt")
    echo "🛑 Prompt 서비스 종료 중..."
    $DOCKER_CMD compose --profile prompt-only down
    ;;
  "rag-reranker")
    echo "🛑 RAG + Reranker 서비스 종료 중..."
    $DOCKER_CMD compose down nginx rag reranker standalone etcd etcd_init minio
    ;;
  "db")
    echo "🛑 데이터베이스 서비스 종료 중..."
    $DOCKER_CMD compose --profile db-only down
    ;;
  "app-only")
    echo "🛑 앱 서비스만 종료 중... (CPU 모드)"
    $DOCKER_CMD compose down nginx rag reranker prompt ollama
    ;;
  "app-only-gpu")
    echo "🛑 앱 서비스만 종료 중... (GPU 모드)"
    $DOCKER_CMD compose down nginx rag reranker prompt ollama-gpu
    ;;
  "ollama")
    echo "🛑 Ollama 서비스 종료 중... (CPU 모드)"
    $DOCKER_CMD compose --profile ollama-only --profile cpu-only down
    ;;
  "ollama-gpu")
    echo "🛑 Ollama 서비스 종료 중... (GPU 모드)"
    $DOCKER_CMD compose --profile gpu-only down
    ;;
  "prompt_ollama")
    echo "🛑 Prompt + Ollama 서비스 종료 중... (CPU 모드)"
    $DOCKER_CMD compose --profile prompt-only --profile ollama-only --profile cpu-only down
    ;;
  "prompt_ollama-gpu")
    echo "🛑 Prompt + Ollama 서비스 종료 중... (GPU 모드)"
    $DOCKER_CMD compose --profile prompt-only --profile gpu-only down
    ;;
  *)
    echo "Usage: $0 {all|all-gpu|rag|reranker|prompt|rag-reranker|db|app-only|app-only-gpu|ollama|ollama-gpu|prompt_ollama|prompt_ollama-gpu}"
    echo "  all        - 모든 서비스 종료 (CPU 모드)"
    echo "  all-gpu    - 모든 서비스 종료 (GPU 모드)"
    echo "  rag          - RAG 서비스만 종료 (DB 포함)"
    echo "  reranker     - Reranker 서비스만 종료"
    echo "  prompt       - Prompt 서비스만 종료"
    echo "  rag-reranker - RAG와 Reranker 서비스 종료 (DB 포함)"
    echo "  db           - 데이터베이스 서비스만 종료"
    echo "  app-only     - 앱 서비스만 종료 (RAG, Reranker, Prompt, Ollama(CPU))"
    echo "  app-only-gpu - 앱 서비스만 종료 (RAG, Reranker, Prompt, Ollama(GPU))"
    echo "  ollama       - Ollama 서비스만 종료 (CPU 모드)"
    echo "  ollama-gpu   - Ollama 서비스만 종료 (GPU 모드)"
    echo "  prompt_ollama - Prompt와 Ollama 서비스 종료 (CPU 모드)"
    echo "  prompt_ollama-gpu - Prompt와 Ollama 서비스 종료 (GPU 모드)"
    exit 1
    ;;
esac

# 소켓 파일 정리
echo "🧹 소켓 파일 정리 중..."
rm -f /tmp/rag.sock /tmp/reranker.sock /tmp/prompt.sock 2>/dev/null || true

echo "=== 종료 완료 ==="
echo "✨ 서비스가 안전하게 종료되었습니다."
echo "💡 도커 이미지와 볼륨은 유지됩니다. 전체 정리를 원하시면 cleanup.sh를 실행하세요." 