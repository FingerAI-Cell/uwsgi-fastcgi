@echo off
setlocal enabledelayedexpansion

echo === 서비스 안전 종료 시작 ===

REM 현재 디렉토리 확인 및 루트 디렉토리로 이동
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\.."
cd /d "%ROOT_DIR%"

REM Docker 실행 확인
docker ps >nul 2>&1
if errorlevel 1 (
    echo Error: Docker가 실행 중이지 않습니다.
    exit /b 1
)

REM 서비스 종료
if "%1"=="" (
    echo Usage: %0 {full^|rag^|reranker^|db^|app-only}
    echo   full     - 모든 서비스 종료
    echo   rag      - RAG 서비스만 종료
    echo   reranker - Reranker 서비스만 종료
    echo   db       - 데이터베이스 서비스만 종료
    echo   app-only - 애플리케이션 서비스만 종료
    exit /b 1
)

if "%1"=="full" (
    echo 모든 서비스 종료 중...
    docker compose --profile full down
) else if "%1"=="rag" (
    echo RAG 서비스 종료 중...
    docker compose --profile rag-only down
) else if "%1"=="reranker" (
    echo Reranker 서비스 종료 중...
    docker compose --profile reranker-only down
) else if "%1"=="db" (
    echo 데이터베이스 서비스 종료 중...
    docker compose --profile db-only down
) else if "%1"=="app-only" (
    echo 애플리케이션 서비스 종료 중...
    docker compose --profile app-only down
) else (
    echo 잘못된 프로파일입니다: %1
    exit /b 1
)

REM 소켓 파일 정리
echo 소켓 파일 정리 중...
del /f /q /tmp\rag.sock 2>nul
del /f /q /tmp\reranker.sock 2>nul

echo === 종료 완료 ===
echo 서비스가 안전하게 종료되었습니다.
echo 도커 이미지와 볼륨은 유지됩니다. 전체 정리를 원하시면 cleanup.bat를 실행하세요. 