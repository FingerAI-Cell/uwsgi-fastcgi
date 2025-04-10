#!/bin/bash

echo "=== 로컬 볼륨 디렉토리 완전 제거 스크립트 ==="

# 현재 디렉토리 확인 및 루트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR" || exit 1

# 경고 메시지
echo "⚠️ 경고: 이 스크립트는 모든 로컬 볼륨 디렉토리와 데이터를 영구적으로 삭제합니다!"
echo "이 작업은 취소할 수 없으며, 모든 저장된 데이터가 손실됩니다."
echo "중요한 데이터는 먼저 백업하세요."
echo "계속하시겠습니까? (y/N)"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "작업이 취소되었습니다."
    exit 0
fi

# 로컬 볼륨 디렉토리 삭제
echo "🗑️ 로컬 볼륨 디렉토리 삭제 중..."
rm -rf ./volumes

# 호스트 내부 볼륨 디렉토리 삭제
echo "🗑️ 내부 볼륨 디렉토리 삭제 중..."
sudo rm -rf /var/lib/milvus-data

# 재생성 - 로컬 디렉토리 (설정 파일과 로그용)
echo "📁 기본 볼륨 디렉토리 구조 재생성 중..."
mkdir -p volumes/logs/{nginx,rag,reranker}

# 재생성 - 호스트 내부 디렉토리
echo "📁 내부 볼륨 디렉토리 구조 재생성 중..."
sudo mkdir -p /var/lib/milvus-data/{etcd,minio,milvus,logs/{etcd,minio,milvus}}
sudo chown -R $(whoami):$(whoami) /var/lib/milvus-data
chmod -R 700 /var/lib/milvus-data/etcd

echo "=== 작업 완료 ==="
echo "✅ 모든 볼륨 디렉토리가 초기화되었습니다."
echo "시스템을 다시 시작하면 깨끗한 상태로 시작됩니다." 