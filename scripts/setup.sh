#!/bin/bash

# 환경 변수 로드
set -a
source .env
set +a

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

# 볼륨 디렉토리 생성
mkdir -p volumes/etcd volumes/minio volumes/milvus
mkdir -p volumes/logs/{etcd,minio,milvus,nginx,rag,reranker}

# nginx 설정 파일 관리
setup_nginx() {
    local mode=$1
    echo "nginx 설정 파일 설정 중 ($mode)..."
    
    # conf.d 디렉토리 초기화
    rm -f nginx/conf.d/*.conf
    
    # 모드에 따른 설정 파일 복사
    case "$mode" in
        "full")
            cp nginx/conf.d.backup/rag.conf nginx/conf.d/
            cp nginx/conf.d.backup/reranker.conf nginx/conf.d/
            ;;
        "rag")
            cp nginx/conf.d.backup/rag.conf nginx/conf.d/
            ;;
        "reranker")
            cp nginx/conf.d.backup/reranker.conf nginx/conf.d/
            ;;
    esac
    
    # 소켓 파일 권한 설정
    touch /tmp/rag.sock /tmp/reranker.sock
    chmod 666 /tmp/rag.sock /tmp/reranker.sock
    
    # nginx 재시작
    if $DOCKER_CMD ps | grep -q milvus-nginx; then
        echo "nginx 재시작 중..."
        $DOCKER_CMD restart milvus-nginx
    fi
}

# FastCGI 환경 변수 설정
export UWSGI_PROTOCOL=fastcgi
export UWSGI_CHMOD_SOCKET=666

# 서비스 시작
case "$1" in
  "full")
    echo "Starting all services..."
    setup_nginx "full"
    docker compose --profile full up -d
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
    ;;
  "rag")
    echo "Starting RAG service..."
    setup_nginx "rag"
    docker compose --profile rag-only up -d
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
    ;;
  "reranker")
    echo "Starting Reranker service..."
    setup_nginx "reranker"
    docker compose --profile reranker-only up -d
    ;;
  *)
    echo "Usage: $0 {full|rag|reranker}"
    echo "  full     - Start all services"
    echo "  rag      - Start RAG service only"
    echo "  reranker - Start Reranker service only"
    exit 1
    ;;
esac

# 서비스 상태 확인
echo "서비스 상태 확인 중..."
$DOCKER_CMD ps | grep -E 'milvus|api-gateway|unified-nginx'

echo "=== 셋업 완료 ==="
echo "시스템이 가동되었습니다. 다음 URL로 접근할 수 있습니다:"
if [ "$1" = "full" ]; then
    echo "- RAG 서비스: http://localhost/rag/"
    echo "- Reranker 서비스: http://localhost/reranker/"
    echo "- 통합 API: http://localhost/api/enhanced-search?query_text=검색어"
elif [ "$1" = "rag" ]; then
    echo "- RAG 서비스: http://localhost/"
elif [ "$1" = "reranker" ]; then
    echo "- Reranker 서비스: http://localhost/"
    echo "- API: http://localhost/api/enhanced-search?query_text=검색어"
fi
echo "- Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)" 