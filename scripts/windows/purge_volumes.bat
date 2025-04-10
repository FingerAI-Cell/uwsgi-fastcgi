@echo off
setlocal enabledelayedexpansion

echo === 로컬 볼륨 디렉토리 완전 제거 스크립트 ===

REM 현재 디렉토리 확인 및 루트 디렉토리로 이동
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\.."
cd /d "%ROOT_DIR%"

REM 경고 메시지
echo ⚠️ 경고: 이 스크립트는 모든 로컬 볼륨 디렉토리와 데이터를 영구적으로 삭제합니다!
echo 이 작업은 취소할 수 없으며, 모든 저장된 데이터가 손실됩니다.
set /p response=계속하시겠습니까? (y/N): 
if /i not "%response%"=="y" (
    echo 작업이 취소되었습니다.
    exit /b 0
)

REM 로컬 볼륨 디렉토리 삭제
echo 🗑️ 로컬 볼륨 디렉토리 삭제 중...
rmdir /s /q volumes 2>nul

REM WSL 내부 볼륨 디렉토리 삭제 안내
echo ⚠️ 주의: WSL 또는 VM 내부 볼륨 디렉토리도 정리해야 합니다.
echo WSL 환경을 사용하는 경우 다음 명령을 WSL 터미널에서 실행하세요:
echo wsl -d Ubuntu sudo rm -rf /var/lib/milvus-data
echo wsl -d Ubuntu sudo mkdir -p /var/lib/milvus-data/{etcd,minio,milvus,logs/{etcd,minio,milvus}}
echo wsl -d Ubuntu sudo chown -R $(whoami):$(whoami) /var/lib/milvus-data
echo wsl -d Ubuntu chmod -R 700 /var/lib/milvus-data/etcd

REM 재생성 (로컬 디렉토리)
echo 📁 기본 볼륨 디렉토리 구조 재생성 중...
mkdir volumes\logs\nginx 2>nul
mkdir volumes\logs\rag 2>nul
mkdir volumes\logs\reranker 2>nul

echo === 작업 완료 ===
echo ✅ 로컬 볼륨 디렉토리가 초기화되었습니다.
echo ⚠️ WSL/VM 내부 디렉토리는 별도로 정리해야 합니다.
echo 시스템을 다시 시작하면 깨끗한 상태로 시작됩니다. 