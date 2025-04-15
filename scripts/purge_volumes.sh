#!/bin/bash

echo "=== 로컬 볼륨 디렉토리 완전 제거 스크립트 ==="

# 현재 디렉토리 확인 및 루트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR" || exit 1

# 설정 파일 경로 (절대 경로 사용)
CONFIG_DIR="$ROOT_DIR/config"
CONFIG_FILE="$CONFIG_DIR/storage.json"

# config 디렉토리 생성
mkdir -p "$CONFIG_DIR"

# Milvus 데이터 경로 확인
DEFAULT_MILVUS_PATH="/var/lib/milvus-data"
if [ -f "$CONFIG_FILE" ]; then
    echo "설정 파일 ($CONFIG_FILE) 을 읽었습니다."
    echo "설정 파일 내용:"
    cat "$CONFIG_FILE"
    echo
    
    # jq 대신 grep과 sed를 사용하여 milvus_data_path 값을 추출
    STORED_PATH=$(grep -o '"milvus_data_path"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_FILE" | sed 's/.*"milvus_data_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
    if [ ! -z "$STORED_PATH" ]; then
        echo "설정된 경로를 찾았습니다: $STORED_PATH"
        MILVUS_PATH=$STORED_PATH
    else
        echo "설정된 경로를 찾을 수 없어 기본값을 사용합니다."
        MILVUS_PATH=$DEFAULT_MILVUS_PATH
    fi
else
    echo "설정 파일이 없어 기본값을 사용합니다: $DEFAULT_MILVUS_PATH"
    MILVUS_PATH=$DEFAULT_MILVUS_PATH
    echo
fi

echo "⚠️ 경고: 이 스크립트는 모든 로컬 볼륨 디렉토리와 데이터를 영구적으로 삭제합니다!"
echo "이 작업은 취소할 수 없으며, 모든 저장된 데이터가 손실됩니다."
echo "중요한 데이터는 먼저 백업하세요."
echo
echo "다음 위치의 데이터가 삭제됩니다:"
echo "1. 로컬 볼륨 디렉토리: ./volumes/"
echo "2. Milvus 데이터 디렉토리: $MILVUS_PATH"
echo
echo "계속하시겠습니까? (y/N)"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "작업이 취소되었습니다."
    exit 0
fi

# 로컬 볼륨 디렉토리 삭제
echo "🗑️ 로컬 볼륨 디렉토리 삭제 중..."
rm -rf ./volumes

# Milvus 데이터 디렉토리 삭제
echo "🗑️ Milvus 데이터 디렉토리 삭제 중..."
if [ -d "$MILVUS_PATH" ]; then
    sudo rm -rf "$MILVUS_PATH"/*
    echo "✓ $MILVUS_PATH 디렉토리의 내용이 삭제되었습니다."
else
    echo "! $MILVUS_PATH 디렉토리가 존재하지 않습니다."
fi

# 재생성 - 로컬 디렉토리 (설정 파일과 로그용)
echo "📁 기본 볼륨 디렉토리 구조 재생성 중..."
mkdir -p volumes/logs/{nginx,rag,reranker,prompt}
echo "✓ 로컬 로그 디렉토리가 재생성되었습니다."

# 재생성 - Milvus 데이터 디렉토리
echo "📁 Milvus 데이터 디렉토리 구조 재생성 중..."
sudo mkdir -p "$MILVUS_PATH"/{etcd,minio,milvus,logs/{etcd,minio,milvus}}
sudo chown -R $(whoami):$(whoami) "$MILVUS_PATH"
sudo chmod -R 700 "$MILVUS_PATH/etcd"
echo "✓ Milvus 데이터 디렉토리가 재생성되었습니다."

echo "=== 작업 완료 ==="
echo "✅ 모든 볼륨 디렉토리가 초기화되었습니다."
echo "시스템을 다시 시작하면 깨끗한 상태로 시작됩니다." 