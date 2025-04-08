#!/bin/bash

# 스크립트 시작 메시지 출력
echo "=== RAG 시스템 셋업 시작 ==="

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

# 필요한 Docker 네트워크 생성 (이미 존재하는 경우 무시)
echo "Docker 네트워크 생성 중..."
$DOCKER_CMD network create rag_network 2>/dev/null || echo "rag_network가 이미 존재합니다."

# 서비스 시작
echo "통합 서비스 시작 중..."
if [[ "$DOCKER_CMD" == "sudo docker" ]]; then
    sudo docker compose -f ./docker-compose.yml up -d
else
    docker compose -f ./docker-compose.yml up -d
fi

# 서비스 상태 확인
echo "서비스 상태 확인 중..."
$DOCKER_CMD ps | grep -E 'milvus|api-gateway|unified-nginx'

echo "=== 셋업 완료 ==="
echo "시스템이 가동되었습니다. 다음 URL로 접근할 수 있습니다:"
echo "- RAG 서비스: http://localhost/"
echo "- Reranker 서비스: http://localhost/reranker/ (통합 서비스 내 경로)"
echo "- 통합 API: http://localhost/api/enhanced-search?query_text=검색어"
echo "- Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)"
echo "참고: 개별 실행 시 Reranker 서비스는 http://localhost:8080/reranker/ 로 접근합니다." 