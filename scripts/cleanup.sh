#!/bin/bash

echo "=== 도커 시스템 전체 정리 시작 ==="

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
        echo "다음 명령어를 실행하면 sudo 없이 Docker를 사용할 수 있습니다:"
        echo "  sudo usermod -aG docker $USER"
        echo "  (위 명령 실행 후 로그아웃 후 다시 로그인해야 합니다)"
    fi
fi

# 경고 메시지
echo "⚠️ 경고: 이 스크립트는 모든 도커 리소스(컨테이너, 이미지, 볼륨, 네트워크)를 삭제합니다!"
echo "계속하시겠습니까? (y/N)"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "작업이 취소되었습니다."
    exit 0
fi

echo "🛑 모든 컨테이너 중지 및 삭제..."
$DOCKER_CMD stop $($DOCKER_CMD ps -aq) 2>/dev/null || true
$DOCKER_CMD rm $($DOCKER_CMD ps -aq) 2>/dev/null || true

echo "🗑️ 모든 이미지 삭제..."
$DOCKER_CMD rmi -f $($DOCKER_CMD images -q) 2>/dev/null || true

echo "🗑️ 모든 볼륨 삭제..."
$DOCKER_CMD volume rm $($DOCKER_CMD volume ls -q) 2>/dev/null || true

echo "🗑️ 사용자 정의 네트워크 삭제..."
$DOCKER_CMD network rm $($DOCKER_CMD network ls --filter "type=custom" -q) 2>/dev/null || true

echo "🧹 빌드 캐시 정리..."
$DOCKER_CMD builder prune -a --force

# 소켓 파일 정리
echo "🧹 소켓 파일 정리..."
for sock in /tmp/rag.sock /tmp/reranker.sock; do
    if [ -S "$sock" ]; then
        echo "소켓 파일 삭제 중: $sock"
        rm -f "$sock"
    fi
done

# nginx 설정 파일 정리
echo "🧹 nginx 설정 파일 정리..."
rm -f nginx/conf.d/*.conf

echo "=== 정리 완료 ==="
echo "✨ 모든 도커 리소스가 정리되었습니다." 