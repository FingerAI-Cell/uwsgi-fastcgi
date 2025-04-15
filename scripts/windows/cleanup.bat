@echo off
setlocal enabledelayedexpansion

echo === 도커 시스템 전체 정리 시작 ===

REM 현재 디렉토리 확인 및 루트 디렉토리로 이동
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\.."
cd /d "%ROOT_DIR%"

REM Docker 실행 확인
docker ps >nul 2>&1
if errorlevel 1 (
    echo Error: Docker가 실행 중이지 않습니다.
    echo Docker Desktop을 실행하고 다시 시도하세요.
    exit /b 1
)

REM 경고 메시지
echo ⚠️ 경고: 이 스크립트는 모든 도커 리소스(컨테이너, 이미지, 볼륨, 네트워크)를 삭제합니다!
set /p response=계속하시겠습니까? (y/N): 
if /i not "%response%"=="y" (
    echo 작업이 취소되었습니다.
    exit /b 0
)

echo 🛑 모든 컨테이너 중지 및 삭제...
for /f "tokens=*" %%a in ('docker ps -aq') do (
    docker stop %%a 2>nul
    docker rm %%a 2>nul
)

echo 🗑️ 모든 이미지 삭제...
for /f "tokens=*" %%a in ('docker images -q') do (
    docker rmi -f %%a 2>nul
)

echo 🗑️ 모든 볼륨 삭제...
for /f "tokens=*" %%a in ('docker volume ls -q') do (
    docker volume rm %%a 2>nul
)

echo 🗑️ 사용자 정의 네트워크 삭제...
for /f "tokens=*" %%a in ('docker network ls --filter "type=custom" -q') do (
    docker network rm %%a 2>nul
)

echo 🧹 빌드 캐시 정리...
docker builder prune -a --force

REM 소켓 파일 정리
echo 🧹 소켓 파일 정리...
del /f /q /tmp\rag.sock 2>nul
del /f /q /tmp\reranker.sock 2>nul
del /f /q /tmp\prompt.sock 2>nul

REM nginx 설정 파일 정리
echo 🧹 nginx 설정 파일 정리...
REM server_base.conf는 보존하고 다른 conf 파일만 삭제
for %%f in (nginx\conf.d\*.conf) do (
    if /i not "%%~nxf"=="server_base.conf" (
        del /f /q "%%f" 2>nul
    )
)
del /f /q nginx\locations-enabled\*.conf 2>nul

echo === 정리 완료 ===
echo ✨ 모든 도커 리소스가 정리되었습니다. 