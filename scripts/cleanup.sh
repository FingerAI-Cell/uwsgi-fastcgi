#!/bin/bash

# 스크립트 시작 메시지 출력
echo "=== RAG 시스템 완전 정리 시작 ==="

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

# 경고 메시지 표시 및 확인
echo "경고: 이 스크립트는 모든 서비스를 중지하고, 관련 볼륨과 네트워크를 삭제합니다."
echo "모든 데이터가 삭제될 수 있습니다. 계속하시겠습니까? (y/n)"
read -r CONFIRM

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "작업이 취소되었습니다."
    exit 0
fi

# 서비스 중지 및 볼륨 삭제
echo "모든 서비스 중지 및 볼륨 삭제 중..."
if [[ "$DOCKER_CMD" == "sudo docker" ]]; then
    sudo docker compose -f ./docker-compose.yml down -v
    sudo docker compose -f ./docker-compose-rag.yml down -v
    sudo docker compose -f ./docker-compose-reranker.yml down -v
else
    docker compose -f ./docker-compose.yml down -v
    docker compose -f ./docker-compose-rag.yml down -v
    docker compose -f ./docker-compose-reranker.yml down -v
fi

# 사용자 정의 네트워크 삭제 (기본 네트워크 제외)
echo "Docker 사용자 정의 네트워크 삭제 중..."
if [[ "$DOCKER_CMD" == "sudo docker" ]]; then
    custom_networks=$(sudo docker network ls --filter "type=custom" -q)
    if [ -n "$custom_networks" ]; then
        sudo docker network rm $custom_networks 2>/dev/null || echo "일부 네트워크는 사용 중이거나 이미 삭제되었습니다."
    else
        echo "삭제할 사용자 정의 네트워크가 없습니다."
    fi
else
    custom_networks=$(docker network ls --filter "type=custom" -q)
    if [ -n "$custom_networks" ]; then
        docker network rm $custom_networks 2>/dev/null || echo "일부 네트워크는 사용 중이거나 이미 삭제되었습니다."
    else
        echo "삭제할 사용자 정의 네트워크가 없습니다."
    fi
fi

echo "=== 정리 완료 ==="
echo "모든 서비스, 볼륨, 네트워크가 삭제되었습니다." 