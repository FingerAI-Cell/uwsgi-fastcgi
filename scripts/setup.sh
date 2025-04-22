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

# 설정 파일 경로 (절대 경로 사용)
CONFIG_DIR="$ROOT_DIR/config"
CONFIG_FILE="$CONFIG_DIR/storage.json"

# 설정 파일 디렉토리 생성
mkdir -p "$CONFIG_DIR"

# 기본 경로
DEFAULT_MILVUS_PATH="/var/lib/milvus-data"
CURRENT_MILVUS_PATH=""

# 설정 파일이 있으면 읽기
if [ -f "$CONFIG_FILE" ]; then
    # jq 대신 grep과 sed를 사용하여 milvus_data_path 값을 추출
    STORED_PATH=$(grep -o '"milvus_data_path"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_FILE" | sed 's/.*"milvus_data_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
    if [ ! -z "$STORED_PATH" ]; then
        CURRENT_MILVUS_PATH=$STORED_PATH
    fi
fi

# 사용자 입력 안내
echo "============= Milvus 데이터 경로 설정 ============="
echo "현재 프로젝트 루트 디렉토리: $(pwd)"
if [ ! -z "$CURRENT_MILVUS_PATH" ]; then
    echo "현재 설정된 경로: $CURRENT_MILVUS_PATH"
fi
echo "다음과 같은 형식의 경로를 입력할 수 있습니다:"
echo "1. 절대 경로 (예: $DEFAULT_MILVUS_PATH)"
echo "2. 프로젝트 루트 기준 상대 경로 (예: ./data/milvus)"
echo "※ 주의: './data/milvus'와 같이 입력하면 '$(pwd)/data/milvus'로 처리됩니다."
echo "※ 권장: 데이터 관리를 위해 절대 경로 사용을 권장합니다."
echo "=================================================="
echo -n "Milvus 데이터 저장 경로를 입력하세요 (기본값: $DEFAULT_MILVUS_PATH): "
read MILVUS_PATH

# 입력이 없으면 기본값 사용
if [ -z "$MILVUS_PATH" ]; then
    echo "기본값을 사용합니다: $DEFAULT_MILVUS_PATH"
    MILVUS_PATH=$DEFAULT_MILVUS_PATH
fi

# 상대 경로를 절대 경로로 변환
if [[ "$MILVUS_PATH" =~ ^\./ ]] || [[ "$MILVUS_PATH" =~ ^\.\./ ]]; then
    # 디렉토리 부분과 파일명 부분 분리
    DIR_PART=$(dirname "$MILVUS_PATH")
    BASE_PART=$(basename "$MILVUS_PATH")
    
    # 상위 디렉토리가 없어도 mkdir로 생성
    mkdir -p "$DIR_PART"
    
    # 절대 경로로 변환 (디렉토리 생성 후)
    ABSOLUTE_DIR=$(cd "$DIR_PART" && pwd)
    if [ $? -eq 0 ]; then
        MILVUS_PATH="$ABSOLUTE_DIR/$BASE_PART"
        echo "상대 경로가 다음 절대 경로로 변환되었습니다: $MILVUS_PATH"
    else
        echo "오류: 디렉토리 생성 또는 접근에 실패했습니다."
        exit 1
    fi
fi

# 경로 유효성 검사
if [[ ! "$MILVUS_PATH" =~ ^/ ]]; then
    echo "오류: 올바르지 않은 경로입니다. 절대 경로(/)나 상대 경로(./ 또는 ../)로 시작해야 합니다."
    exit 1
fi

# 경로 생성
echo "데이터 디렉토리 생성 중..."
if ! mkdir -p "$MILVUS_PATH" 2>/dev/null; then
    echo "오류: 경로를 생성할 수 없습니다. 권한을 확인해주세요."
    exit 1
fi

echo "Milvus 데이터 경로: $MILVUS_PATH"

# 설정 저장
cat > "$CONFIG_FILE" << EOF
{
    "milvus_data_path": "$MILVUS_PATH",
    "created_at": "$(date -Iseconds)",
    "last_modified": "$(date -Iseconds)"
}
EOF

echo "설정 파일 위치: $CONFIG_FILE"

# Milvus 관련 디렉토리 생성
echo "Milvus 데이터 디렉토리 생성 중..."
sudo mkdir -p "$MILVUS_PATH"/{etcd,minio,milvus,logs/{etcd,minio,milvus}}
sudo chmod -R 700 "$MILVUS_PATH/etcd"

# 환경 변수로 export
export MILVUS_DATA_PATH="$MILVUS_PATH"

# 권한 설정 검증
echo "내부 볼륨 권한 설정 확인 중..."
echo "etcd 디렉토리 권한:"
ls -ld "$MILVUS_PATH/etcd"

ETCD_PERM=$(stat -c %a "$MILVUS_PATH/etcd" 2>/dev/null || echo "0")
if [ "$ETCD_PERM" != "700" ]; then
    echo "경고: etcd 디렉토리 권한이 700이 아닙니다. 다시 설정합니다."
    chmod -R 700 "$MILVUS_PATH/etcd"
    ls -ld "$MILVUS_PATH/etcd"
fi

echo "소유권 확인:"
ls -ld "$MILVUS_PATH"

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
    
    # 소켓 파일 권한 설정 (공유 폴더 고려)
    for sock in /tmp/rag.sock /tmp/reranker.sock /tmp/prompt.sock; do
        if [ -S "$sock" ]; then
            chmod 666 "$sock" 2>/dev/null || true
        else
            touch "$sock" 2>/dev/null || true
            chmod 666 "$sock" 2>/dev/null || true
        fi
    done
    
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
    ["rag_reranker-gpu"]="app-only,gpu-only"
    ["prompt"]="prompt-only"
    ["rag"]="rag-only"
    ["reranker"]="reranker-only"
    ["reranker-gpu"]="reranker-only,gpu-only"
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

# Reranker 설정 함수 수정
setup_reranker() {
    local mode=$1
    echo "Reranker 설정 구성 중... (모드: $mode)"
    
    # Config.py 파일 존재 확인
    if [ ! -f "flashrank/Config.py" ]; then
        echo "경고: flashrank/Config.py 파일이 없습니다."
        return 1
    fi
    
    if [ "$mode" = "gpu" ]; then
        # NVIDIA 드라이버 확인
        if ! command -v nvidia-smi &> /dev/null; then
            echo "경고: NVIDIA 드라이버가 설치되어 있지 않습니다."
            echo "GPU 모드를 사용하려면 NVIDIA 드라이버가 필요합니다."
            return 1
        fi
        
        cp reranker/requirements.gpu.txt reranker/requirements.txt
        cp reranker/Dockerfile.gpu reranker/Dockerfile
        
        # Config.py 수정 시도
        if ! sed -i 's/torch_dtype=torch.float32/torch_dtype=torch.float16\n            device_map="auto"/' flashrank/Config.py; then
            echo "경고: Config.py 수정에 실패했습니다."
            return 1
        fi
    else
        cp reranker/requirements.cpu.txt reranker/requirements.txt
        cp reranker/Dockerfile.cpu reranker/Dockerfile
        
        if ! sed -i 's/torch_dtype=torch.float16\n            device_map="auto"/torch_dtype=torch.float32/' flashrank/Config.py; then
            echo "경고: Config.py 수정에 실패했습니다."
            return 1
        fi
    fi
    
    echo "Reranker 설정이 성공적으로 변경되었습니다."
}

# 서비스 시작
case "$1" in
  "all")
    echo "모든 서비스 시작 중... (RAG + Reranker + Prompt + Ollama(CPU) + DB)"
    setup_nginx "all"
    setup_reranker "cpu"
    docker compose --profile all --profile cpu-only up -d
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
    
    # 모델 다운로드 스크립트 실행
    echo "Ollama 모델 다운로드 중..."
    sleep 3
    if docker ps | grep -q milvus-ollama-cpu; then
      chmod 755 "$ROOT_DIR/ollama/init.sh"
      ls -l "$ROOT_DIR/ollama/init.sh"
      docker exec milvus-ollama-cpu /app/init.sh || echo "모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요."
    else
      echo "Ollama 컨테이너가 실행되지 않았습니다. 로그를 확인하세요: docker logs milvus-ollama-cpu"
    fi
    ;;
  "all-gpu")
    echo "모든 서비스 시작 중... (RAG + Reranker + Prompt + Ollama(GPU) + DB)"
    setup_nginx "all"
    setup_reranker "gpu"
    docker compose --profile gpu-only up -d nginx rag reranker prompt ollama-gpu standalone etcd etcd_init minio
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
    
    # 모델 다운로드 스크립트 실행
    echo "Ollama 모델 다운로드 중..."
    sleep 3
    if docker ps | grep -q milvus-ollama-gpu; then
      chmod 755 "$ROOT_DIR/ollama/init.sh"
      ls -l "$ROOT_DIR/ollama/init.sh"
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
    echo "Reranker 서비스만 시작"
    setup_reranker "cpu"
    docker compose --profile reranker-only up -d
    ;;
  "reranker-gpu")
    echo "Reranker 서비스만 시작 (GPU 모드)"
    setup_reranker "gpu"
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
    setup_reranker "cpu"
    docker compose up -d nginx rag reranker standalone etcd etcd_init minio
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
    ;;
  "rag-reranker-gpu")
    echo "RAG + Reranker 서비스 시작 중... (GPU 모드, DB 포함)"
    setup_nginx "rag-reranker"
    setup_reranker "gpu"
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
      chmod 755 "$ROOT_DIR/ollama/init.sh"
      ls -l "$ROOT_DIR/ollama/init.sh"
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
      chmod 755 "$ROOT_DIR/ollama/init.sh"
      ls -l "$ROOT_DIR/ollama/init.sh"
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
      chmod 755 "$ROOT_DIR/ollama/init.sh"
      ls -l "$ROOT_DIR/ollama/init.sh"
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
    chmod 755 "$ROOT_DIR/ollama/init.sh"
      ls -l "$ROOT_DIR/ollama/init.sh"
      docker exec milvus-ollama-cpu /app/init.sh || echo "모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요."
    else
      echo "Ollama 컨테이너가 실행되지 않았습니다. 로그를 확인하세요: docker logs milvus-ollama-cpu"
    fi
    ;;
  "app-only-gpu")
    echo "앱 서비스만 시작 중... (RAG + Reranker + Prompt + Ollama(GPU), DB 제외)"
    setup_nginx "all"
    setup_reranker "gpu"
    docker compose up -d nginx rag reranker prompt ollama-gpu
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4

    # 모델 다운로드 스크립트 실행
    echo "Ollama 모델 다운로드 중..."
    sleep 3
    if docker ps | grep -q milvus-ollama-gpu; then
    chmod 755 "$ROOT_DIR/ollama/init.sh"
      ls -l "$ROOT_DIR/ollama/init.sh"
      docker exec milvus-ollama-cpu /app/init.sh || echo "모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요."
    else
      echo "Ollama 컨테이너가 실행되지 않았습니다. 로그를 확인하세요: docker logs milvus-ollama-gpu"
    fi
    ;;
    ;;
  *)
    echo "Usage: $0 {all|all-gpu|rag|reranker|reranker-gpu|prompt|rag-reranker|rag-reranker-gpu|db|app-only|app-only-gpu|ollama|prompt_ollama|ollama-gpu|prompt_ollama-gpu}"
    echo "  all              - 모든 서비스 시작 (RAG + Reranker + Prompt + Ollama(CPU) + DB)"
    echo "  all-gpu          - 모든 서비스 시작 (RAG + Reranker + Prompt + Ollama(GPU) + DB)"
    echo "  rag              - RAG 서비스만 시작 (DB 포함)"
    echo "  reranker         - Reranker 서비스만 시작 (CPU 모드)"
    echo "  reranker-gpu     - Reranker 서비스만 시작 (GPU 모드)"
    echo "  prompt           - Prompt 서비스만 시작"
    echo "  rag-reranker     - RAG와 Reranker 서비스 시작 (CPU 모드, DB 포함)"
    echo "  rag-reranker-gpu - RAG와 Reranker 서비스 시작 (GPU 모드, DB 포함)"
    echo "  db               - 데이터베이스 서비스만 시작 (Milvus, Etcd, MinIO)"
    echo "  app-only         - 앱 서비스만 시작 (RAG + Reranker + Prompt + Ollama(CPU), DB 제외)"
    echo "  app-only-gpu     - 앱 서비스만 시작 (RAG + Reranker + Prompt + Ollama(GPU), DB 제외)"
    echo "  ollama           - Ollama 서비스만 시작 (CPU 모드)"
    echo "  ollama-gpu       - Ollama 서비스만 시작 (GPU 모드)"
    echo "  prompt_ollama    - Prompt와 Ollama 서비스 조합 (CPU 모드)"
    echo "  prompt_ollama-gpu - Prompt와 Ollama 서비스 조합 (GPU 모드)"
    ;;
esac