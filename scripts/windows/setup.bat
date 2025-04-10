@echo off
setlocal enabledelayedexpansion

:: 스크립트 시작 메시지 출력
echo === RAG 시스템 셋업 시작 ===

:: 현재 디렉토리 확인 및 루트 디렉토리로 이동
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\.."
cd /d "%ROOT_DIR%"

:: Docker 실행 확인
docker ps >nul 2>&1
if errorlevel 1 (
    echo Error: Docker가 실행 중이지 않습니다.
    echo Docker Desktop을 실행하고 다시 시도하세요.
    exit /b 1
)

:: 필요한 Docker 네트워크 생성 (이미 존재하는 경우 무시)
echo Docker 네트워크 생성 중...
docker network create rag_network 2>nul
if errorlevel 1 (
    echo rag_network가 이미 존재합니다.
)

:: 볼륨 디렉토리 생성
mkdir volumes\etcd 2>nul
mkdir volumes\minio 2>nul
mkdir volumes\milvus 2>nul
mkdir volumes\logs\etcd 2>nul
mkdir volumes\logs\minio 2>nul
mkdir volumes\logs\milvus 2>nul
mkdir volumes\logs\nginx 2>nul
mkdir volumes\logs\rag 2>nul
mkdir volumes\logs\reranker 2>nul

:: nginx 설정 파일 관리
:setup_nginx
set "mode=%~1"
echo nginx 설정 파일 설정 중 (%mode%)...

:: locations-enabled 디렉토리 확인
mkdir nginx\locations-enabled 2>nul

:: 모드에 따른 설정 파일 복사
if "%mode%"=="full" (
    :: 둘 다 복사
    copy /y nginx\templates\rag.conf.template nginx\locations-enabled\rag.conf >nul
    copy /y nginx\templates\reranker.conf.template nginx\locations-enabled\reranker.conf >nul
) else if "%mode%"=="rag" (
    :: rag만 복사, reranker는 건드리지 않음
    copy /y nginx\templates\rag.conf.template nginx\locations-enabled\rag.conf >nul
) else if "%mode%"=="reranker" (
    :: reranker만 복사, rag는 건드리지 않음
    copy /y nginx\templates\reranker.conf.template nginx\locations-enabled\reranker.conf >nul
)

:: nginx 재시작
docker ps | findstr "milvus-nginx" >nul
if not errorlevel 1 (
    echo nginx 재시작 중...
    docker restart milvus-nginx
)
goto :eof

:: 서비스 시작
if "%1"=="full" (
    echo Starting all services...
    call :setup_nginx full
    docker compose --profile full up -d
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
) else if "%1"=="rag" (
    echo Starting RAG service...
    call :setup_nginx rag
    docker compose --profile rag-only up -d
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
) else if "%1"=="reranker" (
    echo Starting Reranker service...
    call :setup_nginx reranker
    docker compose --profile reranker-only up -d
) else if "%1"=="db" (
    echo Starting database services only...
    docker compose --profile db-only up -d
) else if "%1"=="app-only" (
    echo Starting application services only...
    call :setup_nginx full
    docker compose --profile app-only up -d
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
) else (
    echo Usage: %0 {full^|rag^|reranker^|db^|app-only}
    echo   full     - Start all services
    echo   rag      - Start RAG service only (includes DB)
    echo   reranker - Start Reranker service only
    echo   db       - Start database services only (Milvus, Etcd, MinIO)
    echo   app-only - Start application services only (RAG, Reranker, Nginx)
    echo             Use this for code changes when DB is already running
    exit /b 1
)

:: 서비스 상태 확인
echo 서비스 상태 확인 중...
docker ps | findstr "milvus api-gateway unified-nginx"

echo === 셋업 완료 ===
echo 시스템이 가동되었습니다. 다음 URL로 접근할 수 있습니다:
if "%1"=="full" (
    echo - RAG 서비스: http://localhost/rag/
    echo - Reranker 서비스: http://localhost/reranker/
    echo - 통합 API: http://localhost/api/enhanced-search?query_text=검색어
) else if "%1"=="rag" (
    echo - RAG 서비스: http://localhost/rag/
) else if "%1"=="reranker" (
    echo - Reranker 서비스: http://localhost/reranker/
    echo - API: http://localhost/api/enhanced-search?query_text=검색어
) else if "%1"=="db" (
    echo - 데이터베이스 서비스만 시작되었습니다. 애플리케이션 서비스는 시작되지 않았습니다.
) else if "%1"=="app-only" (
    echo - RAG 서비스: http://localhost/rag/
    echo - Reranker 서비스: http://localhost/reranker/
    echo - 통합 API: http://localhost/api/enhanced-search?query_text=검색어
)

if "%1"=="full" (
    echo - Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)
) else if "%1"=="rag" (
    echo - Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)
) else if "%1"=="db" (
    echo - Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)
) 