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
  "full")
    echo "🛑 모든 서비스 종료 중..."
    $DOCKER_CMD compose --profile full down
    ;;
  "rag")
    echo "🛑 RAG 서비스 종료 중..."
    $DOCKER_CMD compose --profile rag-only down
    ;;
  "reranker")
    echo "🛑 Reranker 서비스 종료 중..."
    $DOCKER_CMD compose --profile reranker-only down
    ;;
  *)
    echo "Usage: $0 {full|rag|reranker}"
    echo "  full     - 모든 서비스 종료"
    echo "  rag      - RAG 서비스만 종료"
    echo "  reranker - Reranker 서비스만 종료"
    exit 1
    ;;
esac

# 소켓 파일 정리
echo "🧹 소켓 파일 정리 중..."
rm -f /tmp/rag.sock /tmp/reranker.sock 2>/dev/null || true

echo "=== 종료 완료 ==="
echo "✨ 서비스가 안전하게 종료되었습니다."
echo "💡 도커 이미지와 볼륨은 유지됩니다. 전체 정리를 원하시면 cleanup.sh를 실행하세요." 