#!/bin/bash

# 스크립트 시작 메시지 출력
echo "=== Reranker 서비스 종료 시작 ==="

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

# 서비스 중지
echo "Reranker 서비스 중지 중..."
if [[ "$DOCKER_CMD" == "sudo docker" ]]; then
    sudo docker compose -f ./docker-compose-reranker.yml down -v
else
    docker compose -f ./docker-compose-reranker.yml down -v
fi

echo "=== 종료 완료 ==="
echo "Reranker 서비스가 중지되었습니다."
echo "더 완전한 정리를 원하시면 cleanup.sh 스크립트를 실행하세요." 