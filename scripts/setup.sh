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

# 호스트 내부 볼륨 디렉토리 생성 (공유 폴더 권한 문제 해결)
echo "호스트 내부 볼륨 디렉토리 생성 중..."
sudo mkdir -p /var/lib/milvus-data/{etcd,minio,milvus,logs/{etcd,minio,milvus}}
sudo chown -R $(whoami):$(whoami) /var/lib/milvus-data
chmod -R 700 /var/lib/milvus-data/etcd

# 권한 설정 검증
echo "내부 볼륨 권한 설정 확인 중..."
echo "etcd 디렉토리 권한:"
ls -ld /var/lib/milvus-data/etcd

ETCD_PERM=$(stat -c %a /var/lib/milvus-data/etcd 2>/dev/null || echo "0")
if [ "$ETCD_PERM" != "700" ]; then
    echo "경고: etcd 디렉토리 권한이 700이 아닙니다. 다시 설정합니다."
    chmod -R 700 /var/lib/milvus-data/etcd
    ls -ld /var/lib/milvus-data/etcd
fi

echo "소유권 확인:"
ls -ld /var/lib/milvus-data

# 로컬 볼륨 디렉토리 생성 (설정 파일과 로그용)
mkdir -p ./volumes/logs/{nginx,rag,reranker,prompt}

# nginx 설정 파일 관리
setup_nginx() {
    local mode=$1
    echo "nginx 설정 파일 설정 중 ($mode)..."
    
    # locations-enabled 디렉토리 확인
    mkdir -p nginx/locations-enabled
    rm -f nginx/locations-enabled/*.conf
    
    # 모드에 따른 설정 파일 복사
    case "$mode" in
        "all")
            # 모두 복사
            cp nginx/templates/rag.conf.template nginx/locations-enabled/rag.conf
            cp nginx/templates/reranker.conf.template nginx/locations-enabled/reranker.conf
            cp nginx/templates/prompt.conf.template nginx/locations-enabled/prompt.conf
            ;;
        "rag")
            # rag만 복사
            cp nginx/templates/rag.conf.template nginx/locations-enabled/rag.conf
            ;;
        "reranker")
            # reranker만 복사
            cp nginx/templates/reranker.conf.template nginx/locations-enabled/reranker.conf
            ;;
        "prompt")
            # prompt만 복사
            cp nginx/templates/prompt.conf.template nginx/locations-enabled/prompt.conf
            ;;
        "rag-reranker")
            # rag와 reranker만 복사
            cp nginx/templates/rag.conf.template nginx/locations-enabled/rag.conf
            cp nginx/templates/reranker.conf.template nginx/locations-enabled/reranker.conf
            ;;
    esac
    
    # 소켓 파일 권한 설정
    touch /tmp/rag.sock /tmp/reranker.sock /tmp/prompt.sock
    chmod 666 /tmp/rag.sock /tmp/reranker.sock /tmp/prompt.sock
    
    # nginx 재시작
    if $DOCKER_CMD ps | grep -q milvus-nginx; then
        echo "nginx 재시작 중..."
        $DOCKER_CMD restart milvus-nginx
    fi
}

# FastCGI 환경 변수 설정
export UWSGI_PROTOCOL=fastcgi
export UWSGI_CHMOD_SOCKET=666

# 사용 가능한 서비스 목록
prompt_services=(
    "rag_reranker" # RAG + Reranker 서비스 조합
    "prompt" # Prompt 서비스
    "rag" # RAG 서비스
    "reranker" # Reranker 서비스
    "milvus" # Milvus 서비스만
    "ollama" # Ollama 서비스 (CPU)
    "ollama-gpu" # Ollama 서비스 (GPU)
    "prompt_ollama" # Prompt + Ollama 서비스 조합 (CPU)
    "prompt_ollama-gpu" # Prompt + Ollama 서비스 조합 (GPU)
    "all" # 모든 서비스 (CPU 모드)
    "all-gpu" # 모든 서비스 (GPU 모드)
    "app-only" # 앱 서비스만 (CPU 모드)
    "app-only-gpu" # 앱 서비스만 (GPU 모드)
)

# 각 서비스 별 프로필 목록
declare -A profiles=(
    ["rag_reranker"]="app-only"
    ["prompt"]="prompt-only"
    ["rag"]="rag-only"
    ["reranker"]="reranker-only"
    ["milvus"]="db-only"
    ["ollama"]="ollama-only,cpu-only"
    ["ollama-gpu"]="gpu-only"
    ["prompt_ollama"]="prompt-only,ollama-only,cpu-only"
    ["prompt_ollama-gpu"]="prompt-only,gpu-only"
    ["all"]="all,cpu-only"
    ["all-gpu"]="all,gpu-only"
    ["app-only"]="app-only,cpu-only"
    ["app-only-gpu"]="app-only,gpu-only"
)

# 서비스 시작
case "$1" in
  "all")
    echo "모든 서비스 시작 중... (RAG + Reranker + Prompt + Ollama(CPU) + DB)"
    setup_nginx "all"
    docker compose --profile all --profile cpu-only up -d
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
    
    # 모델 다운로드 스크립트 실행
    echo "Ollama 모델 다운로드 중..."
    # 컨테이너가 완전히 시작될 때까지 잠시 대기
    sleep 3
    # 컨테이너가 실행 중인지 확인
    if docker ps | grep -q milvus-ollama-cpu; then
      # 모델 다운로드 시도
      docker exec milvus-ollama-cpu /app/init.sh || echo "모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요."
    else
      echo "Ollama 컨테이너가 실행되지 않았습니다. 로그를 확인하세요: docker logs milvus-ollama-cpu"
    fi
    ;;
  "all-gpu")
    echo "모든 서비스 시작 중... (RAG + Reranker + Prompt + Ollama(GPU) + DB)"
    setup_nginx "all"
    docker compose --profile all --profile gpu-only up -d
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
    
    # 모델 다운로드 스크립트 실행
    echo "Ollama 모델 다운로드 중..."
    sleep 3
    if docker ps | grep -q milvus-ollama-gpu; then
      docker exec milvus-ollama-gpu /app/init.sh || echo "모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요."
    else
      echo "Ollama 컨테이너가 실행되지 않았습니다. 로그를 확인하세요: docker logs milvus-ollama-gpu"
    fi
    ;;
  "rag")
    echo "RAG 서비스 시작 중... (RAG + DB)"
    setup_nginx "rag"
    docker compose --profile rag-only up -d
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
    ;;
  "reranker")
    echo "Reranker 서비스만 시작 중..."
    setup_nginx "reranker"
    docker compose --profile reranker-only up -d
    ;;
  "prompt")
    echo "Prompt 서비스만 시작 중..."
    setup_nginx "prompt"
    docker compose --profile prompt-only up -d
    ;;
  "rag-reranker")
    echo "RAG + Reranker 서비스 시작 중... (DB 포함)"
    setup_nginx "rag-reranker"
    # 커스텀 프로필 대신 실행할 서비스 명시
    docker compose up -d nginx rag reranker standalone etcd etcd_init minio
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
    ;;
  "db")
    echo "데이터베이스 서비스만 시작 중..."
    docker compose --profile db-only up -d
    ;;
  "ollama")
    echo "Ollama 서비스 시작 중... (CPU 모드)"
    docker compose --profile ollama-only --profile cpu-only up -d
    # 모델 다운로드 스크립트 실행
    echo "Ollama 모델 다운로드 중..."
    # 컨테이너가 완전히 시작될 때까지 잠시 대기
    sleep 3
    # 컨테이너가 실행 중인지 확인
    if docker ps | grep -q milvus-ollama-cpu; then
      # 모델 다운로드 시도
      docker exec milvus-ollama-cpu /app/init.sh || echo "모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요."
    else
      echo "Ollama 컨테이너가 실행되지 않았습니다. 로그를 확인하세요: docker logs milvus-ollama-cpu"
    fi
    ;;
  "ollama-gpu")
    echo "Ollama 서비스 시작 중... (GPU 모드)"
    docker compose --profile gpu-only up -d
    # 모델 다운로드 스크립트 실행
    echo "Ollama 모델 다운로드 중..."
    # 컨테이너가 완전히 시작될 때까지 잠시 대기
    sleep 3
    # 컨테이너가 실행 중인지 확인
    if docker ps | grep -q milvus-ollama-gpu; then
      # 모델 다운로드 시도
      docker exec milvus-ollama-gpu /app/init.sh || echo "모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요."
    else
      echo "Ollama 컨테이너가 실행되지 않았습니다. 로그를 확인하세요: docker logs milvus-ollama-gpu"
    fi
    ;;
  "prompt_ollama")
    echo "Prompt와 Ollama 서비스 조합 시작 중... (CPU 모드)"
    setup_nginx "prompt"
    docker compose --profile prompt-only --profile ollama-only --profile cpu-only up -d
    # 모델 다운로드 스크립트 실행
    echo "Ollama 모델 다운로드 중..."
    # 컨테이너가 완전히 시작될 때까지 잠시 대기
    sleep 3
    # 컨테이너가 실행 중인지 확인
    if docker ps | grep -q milvus-ollama-cpu; then
      # 모델 다운로드 시도
      docker exec milvus-ollama-cpu /app/init.sh || echo "모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요."
    else
      echo "Ollama 컨테이너가 실행되지 않았습니다. 로그를 확인하세요: docker logs milvus-ollama-cpu"
    fi
    ;;
  "prompt_ollama-gpu")
    echo "Prompt와 Ollama 서비스 조합 시작 중... (GPU 모드)"
    setup_nginx "prompt"
    docker compose --profile prompt-only --profile gpu-only up -d
    # 모델 다운로드 스크립트 실행
    echo "Ollama 모델 다운로드 중..."
    # 컨테이너가 완전히 시작될 때까지 잠시 대기
    sleep 3
    # 컨테이너가 실행 중인지 확인
    if docker ps | grep -q milvus-ollama-gpu; then
      # 모델 다운로드 시도
      docker exec milvus-ollama-gpu /app/init.sh || echo "모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요."
    else
      echo "Ollama 컨테이너가 실행되지 않았습니다. 로그를 확인하세요: docker logs milvus-ollama-gpu"
    fi
    ;;
  "app-only")
    echo "앱 서비스만 시작 중... (RAG + Reranker + Prompt + Ollama(CPU), DB 제외)"
    setup_nginx "all"
    docker compose up -d nginx rag reranker prompt ollama
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    
    # 모델 다운로드 스크립트 실행
    echo "Ollama 모델 다운로드 중..."
    sleep 3
    if docker ps | grep -q milvus-ollama-cpu; then
      docker exec milvus-ollama-cpu /app/init.sh || echo "모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요."
    else
      echo "Ollama 컨테이너가 실행되지 않았습니다. 로그를 확인하세요: docker logs milvus-ollama-cpu"
    fi
    ;;
  "app-only-gpu")
    echo "앱 서비스만 시작 중... (RAG + Reranker + Prompt + Ollama(GPU), DB 제외)"
    setup_nginx "all"
    docker compose up -d nginx rag reranker prompt ollama-gpu
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    
    # 모델 다운로드 스크립트 실행
    echo "Ollama 모델 다운로드 중..."
    sleep 3
    if docker ps | grep -q milvus-ollama-gpu; then
      docker exec milvus-ollama-gpu /app/init.sh || echo "모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요."
    else
      echo "Ollama 컨테이너가 실행되지 않았습니다. 로그를 확인하세요: docker logs milvus-ollama-gpu"
    fi
    ;;
  *)
    echo "Usage: $0 {all|all-gpu|rag|reranker|prompt|rag-reranker|db|app-only|app-only-gpu|ollama|prompt_ollama|ollama-gpu|prompt_ollama-gpu}"
    echo "  all         - 모든 서비스 시작 (RAG + Reranker + Prompt + Ollama(CPU) + DB)"
    echo "  all-gpu      - 모든 서비스 시작 (RAG + Reranker + Prompt + Ollama(GPU) + DB)"
    echo "  rag          - RAG 서비스만 시작 (DB 포함)"
    echo "  reranker     - Reranker 서비스만 시작"
    echo "  prompt       - Prompt 서비스만 시작"
    echo "  rag-reranker - RAG와 Reranker 서비스 시작 (DB 포함)"
    echo "  db           - 데이터베이스 서비스만 시작 (Milvus, Etcd, MinIO)"
    echo "  app-only     - 앱 서비스만 시작 (RAG + Reranker + Prompt + Ollama(CPU), DB 제외)"
    echo "  app-only-gpu - 앱 서비스만 시작 (RAG + Reranker + Prompt + Ollama(GPU), DB 제외)"
    echo "  ollama       - Ollama 서비스만 시작 (CPU 모드)"
    echo "  ollama-gpu   - Ollama 서비스만 시작 (GPU 모드)"
    echo "  prompt_ollama - Prompt와 Ollama 서비스 조합 (CPU 모드)"
    echo "  prompt_ollama-gpu - Prompt와 Ollama 서비스 조합 (GPU 모드)"
    ;;
esac