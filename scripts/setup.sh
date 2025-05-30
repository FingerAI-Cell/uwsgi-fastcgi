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

# Docker 데몬 설정 확인 및 리소스 위치 출력
echo "============= Docker 데몬 설정 확인 ============="
DOCKER_INFO=$($DOCKER_CMD info --format '{{json .}}')
DOCKER_ROOT=$(echo "$DOCKER_INFO" | grep -o '"DockerRootDir":"[^"]*"' | sed 's/"DockerRootDir":"//;s/"//')
DOCKER_CONFIG_FILE="/etc/docker/daemon.json"

echo "현재 Docker 루트 디렉토리: $DOCKER_ROOT"
echo "Docker 데몬 설정 파일: $DOCKER_CONFIG_FILE"

# 도커 볼륨 위치 확인
VOLUME_PATH="${DOCKER_ROOT}/volumes"
if [ -d "$VOLUME_PATH" ]; then
    echo "Docker 볼륨 저장 위치: $VOLUME_PATH"
else
    echo "Docker 볼륨 저장 위치를 확인할 수 없습니다."
fi

# 도커 이미지 위치 확인
IMAGE_PATH="${DOCKER_ROOT}/image"
if [ -d "$IMAGE_PATH" ]; then
    echo "Docker 이미지 저장 위치: $IMAGE_PATH"
else
    echo "Docker 이미지 저장 위치를 확인할 수 없습니다."
fi

# 도커 컨테이너 위치 확인
CONTAINER_PATH="${DOCKER_ROOT}/containers"
if [ -d "$CONTAINER_PATH" ]; then
    echo "Docker 컨테이너 저장 위치: $CONTAINER_PATH"
else
    echo "Docker 컨테이너 저장 위치를 확인할 수 없습니다."
fi

# 도커 네트워크 위치 확인
NETWORK_PATH="${DOCKER_ROOT}/network"
if [ -d "$NETWORK_PATH" ]; then
    echo "Docker 네트워크 저장 위치: $NETWORK_PATH"
else
    echo "Docker 네트워크 저장 위치를 확인할 수 없습니다."
fi

# 도커 데몬 설정 파일 존재 확인
if [ -f "$DOCKER_CONFIG_FILE" ]; then
    echo "현재 Docker 데몬 설정 내용:"
    cat "$DOCKER_CONFIG_FILE"
    
    # 데이터 저장 위치 변경 여부 질문
    echo ""
    echo "Docker 리소스 저장 위치를 변경하시겠습니까? (y/n): "
    read CHANGE_DOCKER_PATH
    
    if [ "$CHANGE_DOCKER_PATH" = "y" ] || [ "$CHANGE_DOCKER_PATH" = "Y" ]; then
        echo "새로운 Docker 루트 디렉토리 경로를 입력하세요 (절대 경로): "
        read NEW_DOCKER_ROOT
        
        if [ -n "$NEW_DOCKER_ROOT" ]; then
            echo "Docker 루트 디렉토리를 '$NEW_DOCKER_ROOT'로 변경합니다."
            
            # 현재 설정 백업
            if [ -f "$DOCKER_CONFIG_FILE" ]; then
                TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
                BACKUP_FILE="${DOCKER_CONFIG_FILE}.${TIMESTAMP}.bak"
                sudo cp "$DOCKER_CONFIG_FILE" "$BACKUP_FILE"
                echo "기존 설정 파일을 $BACKUP_FILE 로 백업했습니다."
            fi
            
            # 새 설정 파일 생성
            if [ -f "$DOCKER_CONFIG_FILE" ]; then
                # 기존 파일에 data-root 추가/변경
                TMP_FILE=$(mktemp)
                if grep -q '"data-root"' "$DOCKER_CONFIG_FILE"; then
                    # data-root가 이미 있는 경우, 값 변경
                    sudo jq --arg path "$NEW_DOCKER_ROOT" '.["data-root"]=$path' "$DOCKER_CONFIG_FILE" > "$TMP_FILE"
                else
                    # data-root가 없는 경우, 새로 추가
                    sudo jq --arg path "$NEW_DOCKER_ROOT" '. + {"data-root": $path}' "$DOCKER_CONFIG_FILE" > "$TMP_FILE"
                fi
                sudo mv "$TMP_FILE" "$DOCKER_CONFIG_FILE"
            else
                # 새 설정 파일 생성
                sudo mkdir -p /etc/docker
                echo '{
  "data-root": "'$NEW_DOCKER_ROOT'"
}' | sudo tee "$DOCKER_CONFIG_FILE" > /dev/null
            fi
            
            echo "Docker 데몬 설정 파일이 업데이트되었습니다."
            echo "변경사항을 적용하려면 Docker 서비스를 재시작해야 합니다."
            echo "Docker 서비스를 재시작하시겠습니까? (이 작업은 실행 중인 모든 컨테이너를 중지합니다) (y/n): "
            read RESTART_DOCKER
            
            if [ "$RESTART_DOCKER" = "y" ] || [ "$RESTART_DOCKER" = "Y" ]; then
                echo "기존 도커 리소스를 새 위치로 복사합니다. 이 작업은 시간이 걸릴 수 있습니다..."
                
                # 새 위치에 이미 파일이 있는지 확인
                sudo mkdir -p "$NEW_DOCKER_ROOT"
                if [ "$(ls -A "$NEW_DOCKER_ROOT" 2>/dev/null)" ]; then
                    echo "경고: 새 도커 루트 디렉토리($NEW_DOCKER_ROOT)에 이미 파일이 존재합니다."
                    echo "계속 진행하면 기존 파일이 덮어쓰여질 수 있습니다."
                    echo "계속 진행하시겠습니까? (y/n/b - y:진행, n:취소, b:기존 파일 백업 후 진행): "
                    read OVERWRITE_CHOICE
                    
                    if [ "$OVERWRITE_CHOICE" = "n" ] || [ "$OVERWRITE_CHOICE" = "N" ]; then
                        echo "작업을 취소합니다. 도커 설정은 변경되었지만 데이터는 복사되지 않았습니다."
                        echo "변경사항을 적용하려면 Docker 서비스를 수동으로 재시작해 주세요."
                        return
                    elif [ "$OVERWRITE_CHOICE" = "b" ] || [ "$OVERWRITE_CHOICE" = "B" ]; then
                        BACKUP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
                        BACKUP_DIR="${NEW_DOCKER_ROOT}_backup_${BACKUP_TIMESTAMP}"
                        echo "기존 파일을 $BACKUP_DIR 로 백업합니다..."
                        sudo mv "$NEW_DOCKER_ROOT" "$BACKUP_DIR"
                        sudo mkdir -p "$NEW_DOCKER_ROOT"
                        echo "백업 완료. 계속 진행합니다."
                    fi
                    # else (OVERWRITE_CHOICE = y): 그냥 진행
                fi
                
                # 기존 데이터 복사 (볼륨, 이미지, 컨테이너 등)
                echo "도커 데이터를 복사합니다. 이 작업은 데이터 양에 따라 시간이 오래 걸릴 수 있습니다..."
                sudo rsync -av --progress "$DOCKER_ROOT/" "$NEW_DOCKER_ROOT/"
                
                echo "Docker 서비스를 재시작합니다..."
                if command -v systemctl > /dev/null 2>&1; then
                    sudo systemctl restart docker
                elif command -v service > /dev/null 2>&1; then
                    sudo service docker restart
                else
                    echo "Docker 서비스를 자동으로 재시작할 수 없습니다. 수동으로 재시작해 주세요."
                fi
                
                echo "Docker 서비스가 재시작되었습니다."
                echo "새 Docker 루트 디렉토리: $NEW_DOCKER_ROOT"
            else
                echo "Docker 설정이 변경되었지만 아직 적용되지 않았습니다."
                echo "변경사항을 적용하려면 Docker 서비스를 수동으로 재시작해 주세요."
            fi
        else
            echo "유효한 경로가 입력되지 않았습니다. 기존 설정을 유지합니다."
        fi
    else
        echo "Docker 리소스 저장 위치 변경을 건너뜁니다."
    fi
else
    echo "Docker 데몬 설정 파일($DOCKER_CONFIG_FILE)이 존재하지 않습니다."
    echo "새로운 Docker 데몬 설정 파일을 생성하고 리소스 저장 위치를 변경하시겠습니까? (y/n): "
    read CREATE_CONFIG
    
    if [ "$CREATE_CONFIG" = "y" ] || [ "$CREATE_CONFIG" = "Y" ]; then
        echo "새로운 Docker 루트 디렉토리 경로를 입력하세요 (절대 경로): "
        read NEW_DOCKER_ROOT
        
        if [ -n "$NEW_DOCKER_ROOT" ]; then
            echo "Docker 루트 디렉토리를 '$NEW_DOCKER_ROOT'로 설정합니다."
            
            # 새 설정 파일 생성
            sudo mkdir -p /etc/docker
            echo '{
  "data-root": "'$NEW_DOCKER_ROOT'"
}' | sudo tee "$DOCKER_CONFIG_FILE" > /dev/null
            
            echo "Docker 데몬 설정 파일이 생성되었습니다."
            echo "변경사항을 적용하려면 Docker 서비스를 재시작해야 합니다."
            echo "Docker 서비스를 재시작하시겠습니까? (이 작업은 실행 중인 모든 컨테이너를 중지합니다) (y/n): "
            read RESTART_DOCKER
            
            if [ "$RESTART_DOCKER" = "y" ] || [ "$RESTART_DOCKER" = "Y" ]; then
                echo "Docker 서비스를 재시작합니다..."
                if command -v systemctl > /dev/null 2>&1; then
                    sudo systemctl restart docker
                elif command -v service > /dev/null 2>&1; then
                    sudo service docker restart
                else
                    echo "Docker 서비스를 자동으로 재시작할 수 없습니다. 수동으로 재시작해 주세요."
                fi
                
                echo "Docker 서비스가 재시작되었습니다."
                echo "새 Docker 루트 디렉토리: $NEW_DOCKER_ROOT"
            else
                echo "Docker 설정이 변경되었지만 아직 적용되지 않았습니다."
                echo "변경사항을 적용하려면 Docker 서비스를 수동으로 재시작해 주세요."
            fi
        else
            echo "유효한 경로가 입력되지 않았습니다. 기본 설정을 유지합니다."
        fi
    else
        echo "Docker 리소스 저장 위치 변경을 건너뜁니다."
    fi
fi
echo "=================================================="

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
mkdir -p ./volumes/logs/{nginx,rag,reranker,prompt,vision}

# nginx 설정 파일 관리
setup_nginx() {
    local mode=$1
    local stats_enabled=${2:-false}
    echo "nginx 설정 파일 설정 중 ($mode)..."
    
    # locations-enabled 디렉토리 확인
    mkdir -p nginx/locations-enabled
    rm -f nginx/locations-enabled/*.conf
    
    # 통계 수집 활성화 옵션
    if [ "$stats_enabled" = "true" ]; then
        echo "통계 수집 기능을 활성화합니다..."
        cp nginx/templates/stats.conf.template nginx/locations-enabled/stats.conf
    fi
    
    # 모드에 따른 설정 파일 복사
    case "$mode" in
        "all")
            # 모두 복사
            cp nginx/templates/rag.conf.template nginx/locations-enabled/rag.conf
            cp nginx/templates/reranker.conf.template nginx/locations-enabled/reranker.conf
            cp nginx/templates/prompt.conf.template nginx/locations-enabled/prompt.conf
            cp nginx/templates/vision.conf.template nginx/locations-enabled/vision.conf
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
        "vision")
            # vision만 복사
            cp nginx/templates/vision.conf.template nginx/locations-enabled/vision.conf
            ;;
    esac
    
    # 소켓 파일 권한 설정 (공유 폴더 고려)
    for sock in /tmp/rag.sock /tmp/reranker.sock /tmp/prompt.sock /tmp/vision.sock; do
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
    "vision"          # Vision 서비스
    "vision-ollama"   # Vision + Ollama 서비스 (CPU)
    "vision-ollama-gpu" # Vision + Ollama 서비스 (GPU)
    "rag_reranker"    # RAG + Reranker 서비스 조합
    "prompt"          # Prompt 서비스
    "rag"            # RAG 서비스
    "reranker"        # Reranker 서비스
    "milvus"          # Milvus 서비스만
    "ollama"          # Ollama 서비스 (CPU)
    "ollama-gpu"      # Ollama 서비스 (GPU)
    "prompt_ollama"   # Prompt + Ollama 서비스 조합 (CPU)
    "prompt_ollama-gpu" # Prompt + Ollama 서비스 조합 (GPU)
    "all"             # 모든 서비스 (CPU 모드)
    "all-gpu"         # 모든 서비스 (GPU 모드)
    "app-only"        # 앱 서비스만 (CPU 모드)
    "app-only-gpu"    # 앱 서비스만 (GPU 모드)
)

# 각 서비스 별 프로필 목록
declare -A profiles=(
    ["vision"]="vision-only"
    ["vision-gpu"]="vision-only,gpu-only"
    ["vision-ollama"]="vision-only,ollama-only,cpu-only"
    ["vision-ollama-gpu"]="vision-only,gpu-only"
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

# 서비스 설명 목록
declare -A service_descriptions=(
    ["all"]="모든 서비스 시작 (RAG + Reranker + Prompt + Ollama(CPU) + DB + Vision)"
    ["all-gpu"]="모든 서비스 시작 (RAG + Reranker + Prompt + Ollama(GPU) + DB + Vision)"
    ["rag"]="RAG 서비스만 시작 (DB 포함)"
    ["reranker"]="Reranker 서비스만 시작 (CPU 모드)"
    ["reranker-gpu"]="Reranker 서비스만 시작 (GPU 모드)"
    ["prompt"]="Prompt 서비스만 시작"
    ["rag-reranker"]="RAG와 Reranker 서비스 시작 (CPU 모드, DB 포함)"
    ["rag-reranker-gpu"]="RAG와 Reranker 서비스 시작 (GPU 모드, DB 포함)"
    ["db"]="데이터베이스 서비스만 시작 (Milvus, Etcd, MinIO)"
    ["app-only"]="앱 서비스만 시작 (RAG + Reranker + Prompt + Ollama(CPU) + Vision, DB 제외)"
    ["app-only-gpu"]="앱 서비스만 시작 (RAG + Reranker + Prompt + Ollama(GPU) + Vision, DB 제외)"
    ["ollama"]="Ollama 서비스만 시작 (CPU 모드)"
    ["ollama-gpu"]="Ollama 서비스만 시작 (GPU 모드)"
    ["prompt_ollama"]="Prompt와 Ollama 서비스 조합 (CPU 모드)"
    ["prompt_ollama-gpu"]="Prompt와 Ollama 서비스 조합 (GPU 모드)"
    ["vision"]="Vision 서비스만 시작"
    ["vision-ollama"]="Vision과 Ollama 서비스 조합 (CPU 모드)"
    ["vision-ollama-gpu"]="Vision과 Ollama 서비스 조합 (GPU 모드)"
)

# 서비스 구성별 필요 컨테이너 매핑
declare -A service_containers=(
    ["all"]="nginx rag reranker prompt ollama standalone etcd etcd_init minio vision"
    ["all-gpu"]="nginx rag reranker prompt ollama-gpu standalone etcd etcd_init minio vision"
    ["rag"]="nginx rag standalone etcd etcd_init minio"
    ["reranker"]="nginx reranker"
    ["prompt"]="nginx prompt"
    ["rag-reranker"]="nginx rag reranker standalone etcd etcd_init minio"
    ["db"]="standalone etcd etcd_init minio"
    ["app-only"]="nginx rag reranker prompt ollama vision"
    ["app-only-gpu"]="nginx rag reranker prompt ollama-gpu vision"
    ["ollama"]="ollama"
    ["ollama-gpu"]="ollama-gpu"
    ["prompt_ollama"]="nginx prompt ollama"
    ["prompt_ollama-gpu"]="nginx prompt ollama-gpu"
    ["vision"]="nginx vision"
    ["vision-ollama"]="nginx vision ollama"
    ["vision-ollama-gpu"]="nginx vision ollama-gpu"
)

# nginx 설정 모드 매핑
declare -A nginx_modes=(
    ["all"]="all"
    ["all-gpu"]="all"
    ["rag"]="rag"
    ["reranker"]="reranker"
    ["prompt"]="prompt"
    ["rag-reranker"]="rag-reranker"
    ["rag-reranker-gpu"]="rag-reranker"
    ["app-only"]="all"
    ["app-only-gpu"]="all"
    ["prompt_ollama"]="prompt"
    ["prompt_ollama-gpu"]="prompt"
    ["vision"]="vision"
    ["vision-ollama"]="vision"
    ["vision-ollama-gpu"]="vision"
)

# Reranker 설정 함수
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

# 공통 함수: RAG 컨테이너 numpy 패키지 수정
fix_rag_numpy() {
    if docker ps | grep -q milvus-rag; then
        echo "RAG 컨테이너의 numpy 패키지 수정 중..."
        docker exec -it milvus-rag pip uninstall numpy -y
        docker exec -it milvus-rag pip install numpy==1.24.4
    fi
}

# 공통 함수: Ollama 모델 다운로드
download_ollama_models() {
    local container_name=$1
    echo "Ollama 모델 다운로드 중..."
    sleep 3
    if docker ps | grep -q $container_name; then
        chmod 755 "$ROOT_DIR/ollama/init.sh"
        ls -l "$ROOT_DIR/ollama/init.sh"
        docker exec $container_name /app/init.sh || echo "모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요."
    else
        echo "Ollama 컨테이너가 실행되지 않았습니다. 로그를 확인하세요: docker logs $container_name"
    fi
}

# RAG 모델 다운로드 함수
download_rag_model() {
    # 모델 저장 디렉토리 확인
    MODEL_DIR="$ROOT_DIR/models"
    MODEL_PATH="$MODEL_DIR/bge-m3"
    echo "모델 디렉토리 확인: $MODEL_PATH"
    
    # 필수 파일 존재 확인
    REQUIRED_FILES=(
        "pytorch_model.bin"
        "colbert_linear.pt"
        "sentencepiece.bpe.model"
        "sparse_linear.pt"
        "tokenizer.json"
        "config.json"
        "tokenizer_config.json"
        "special_tokens_map.json"
    )
    
    MISSING_FILES=()
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$MODEL_PATH/$file" ]; then
            MISSING_FILES+=("$file")
        fi
    done
    
    if [ ${#MISSING_FILES[@]} -ne 0 ]; then
        echo "오류: 다음 필수 모델 파일이 누락되었습니다:"
        printf '%s\n' "${MISSING_FILES[@]}"
        echo "모델 파일을 올바른 위치에 배치했는지 확인해주세요."
        echo "대용량 파일들은 수동으로 전송해야 합니다."
        exit 1
    fi
    
    echo "모델 파일 확인 완료: $MODEL_PATH"
    # 권한 설정
    chmod -R 755 "$MODEL_PATH"
    return 0
}

# 서비스 시작 함수
start_containers() {
    local mode=$1
    local containers=$2
    local use_profile=$3
    local stats_enabled=${4:-false}  # 통계 수집 활성화 여부
    
    echo "${service_descriptions[$mode]}"
    
    # 컨테이너 시작 전에 모델 다운로드
    if [[ "$containers" == *"rag"* ]]; then
        download_rag_model
    fi
    
    # nginx 설정 (통계 수집 옵션 전달)
    if [[ -n "${nginx_modes[$mode]}" ]]; then
        setup_nginx "${nginx_modes[$mode]}" "$stats_enabled"
    fi
    
    # reranker 설정 (GPU 모드이면)
    if [[ "$mode" == *"-gpu" ]] && [[ "$containers" == *"reranker"* ]]; then
        setup_reranker "gpu"
    elif [[ "$containers" == *"reranker"* ]] && [[ "$mode" != "db" ]]; then
        setup_reranker "cpu"
    fi
    
    # 통계 서비스 추가 (활성화 된 경우)
    local stats_containers=""
    if [ "$stats_enabled" = "true" ]; then
        stats_containers="stats-service stats-db"
    fi
    
    # 컨테이너 시작
    if [ "$use_profile" = true ]; then
        # 프로필 사용
        if [ "$mode" = "stats" ]; then
            docker compose --profile stats up -d
        elif [[ "$mode" == *"-gpu" ]]; then
            if [ "$stats_enabled" = "true" ]; then
                docker compose --profile gpu-only --profile stats up -d
            else
                docker compose --profile gpu-only up -d
            fi
        elif [ "$mode" = "db" ]; then
            if [ "$stats_enabled" = "true" ]; then
                docker compose --profile db-only --profile stats up -d
            else
                docker compose --profile db-only up -d
            fi
        elif [ "$mode" = "rag" ]; then
            if [ "$stats_enabled" = "true" ]; then
                docker compose --profile rag-only --profile stats up -d
            else
                docker compose --profile rag-only up -d
            fi
        elif [ "$mode" = "reranker" ]; then
            if [ "$stats_enabled" = "true" ]; then
                docker compose --profile reranker-only --profile stats up -d
            else
                docker compose --profile reranker-only up -d
            fi
        elif [ "$mode" = "prompt" ]; then
            if [ "$stats_enabled" = "true" ]; then
                docker compose --profile prompt-only --profile stats up -d
            else
                docker compose --profile prompt-only up -d
            fi
        elif [ "$mode" = "vision" ]; then
            if [ "$stats_enabled" = "true" ]; then
                docker compose --profile vision-only --profile stats up -d
            else
                docker compose --profile vision-only up -d
            fi
        elif [ "$mode" = "vision-ollama" ]; then
            if [ "$stats_enabled" = "true" ]; then
                docker compose --profile vision-only --profile ollama-only --profile cpu-only --profile stats up -d
            else
                docker compose --profile vision-only --profile ollama-only --profile cpu-only up -d
            fi
        elif [ "$mode" = "vision-ollama-gpu" ]; then
            if [ "$stats_enabled" = "true" ]; then
                docker compose --profile vision-only --profile gpu-only --profile stats up -d
            else
                docker compose --profile vision-only --profile gpu-only up -d
            fi
        elif [ "$mode" = "ollama" ]; then
            if [ "$stats_enabled" = "true" ]; then
                docker compose --profile ollama-only --profile cpu-only --profile stats up -d
            else
                docker compose --profile ollama-only --profile cpu-only up -d
            fi
        elif [ "$mode" = "prompt_ollama" ]; then
            if [ "$stats_enabled" = "true" ]; then
                docker compose --profile prompt-only --profile ollama-only --profile cpu-only --profile stats up -d
            else
                docker compose --profile prompt-only --profile ollama-only --profile cpu-only up -d
            fi
        elif [ "$mode" = "prompt_ollama-gpu" ]; then
            if [ "$stats_enabled" = "true" ]; then
                docker compose --profile prompt-only --profile gpu-only --profile stats up -d
            else
                docker compose --profile prompt-only --profile gpu-only up -d
            fi
        else
            if [ "$stats_enabled" = "true" ]; then
                docker compose --profile all --profile cpu-only --profile stats up -d
            else
                docker compose --profile all --profile cpu-only up -d
            fi
        fi
    else
        # 명시적 컨테이너 지정
        if [ "$stats_enabled" = "true" ]; then
            docker compose up -d $containers $stats_containers
        else
            docker compose up -d $containers
        fi
    fi
    
    # DB 컨테이너가 포함된 경우 재시작 처리
    if [[ "$containers" == *"standalone"* ]] && [[ "$containers" == *"rag"* ]]; then
        echo "DB와 RAG 서비스 동기화를 위해 컨테이너 재시작..."
        sleep 5  # DB 초기화를 위한 대기
        docker restart milvus-standalone milvus-rag
    fi
    
    # Ollama가 포함된 경우 모델 다운로드
    if [[ "$containers" == *"ollama"* ]] || [[ "$mode" == *"-gpu" && "$mode" != "reranker-gpu" && "$mode" != "rag-reranker-gpu" ]]; then
        download_ollama_models $OLLAMA_CONTAINER
    fi
}

# 사용법 출력 함수
print_usage() {
    local modes=""
    for mode in "${!service_descriptions[@]}"; do
        modes="$modes|$mode"
    done
    modes=${modes:1}  # 첫 번째 | 제거
    
    echo "Usage: $0 {$modes}"
    for mode in "${!service_descriptions[@]}"; do
        printf "  %-18s - %s\n" "$mode" "${service_descriptions[$mode]}"
    done
}

# GPU/CPU 모드에 따라 .env 파일 설정
if [[ "$1" == *"-gpu" ]]; then
    echo "[env] GPU 모드로 설정합니다"
    cp .env.gpu .env
    OLLAMA_CONTAINER="milvus-ollama-gpu"
else
    echo "[env] CPU 모드로 설정합니다"
    cp .env.cpu .env
    OLLAMA_CONTAINER="milvus-ollama-cpu"
fi

# 통계 수집 항상 활성화
STATS_ENABLED=true
echo "[options] 통계 수집 기능이 활성화됩니다."

# 서비스 모드 파싱
SERVICE_MODE=""

# 인수 파싱
for arg in "$@"; do
    if [[ -n "${service_descriptions[$arg]}" ]]; then
        SERVICE_MODE="$arg"
        break
    fi
done

# 서비스 시작
if [ -z "$SERVICE_MODE" ]; then
    print_usage
    exit 1
fi

# 컨테이너 시작
if [ -n "${service_containers[$SERVICE_MODE]}" ]; then
    # 명시적 컨테이너 목록이 있는 경우
    start_containers "$SERVICE_MODE" "${service_containers[$SERVICE_MODE]}" false "$STATS_ENABLED"
else
    # 프로필 사용
    start_containers "$SERVICE_MODE" "" true "$STATS_ENABLED"
fi