@echo off
echo === RAG 시스템 완전 정리 시작 ===

:: 경고 메시지 표시 및 확인
echo 경고: 이 스크립트는 모든 서비스를 중지하고, 관련 컨테이너, 이미지, 볼륨, 네트워크를 삭제합니다.
echo 모든 데이터가 삭제될 수 있습니다. 계속하시겠습니까? (Y/N)
choice /C YN /M "선택:"

if errorlevel 2 (
    echo 작업이 취소되었습니다.
    goto End
)

:: 서비스 중지 및 볼륨 삭제
echo 모든 서비스 중지 및 볼륨 삭제 중...
docker compose -f ..\docker-compose.yml down -v
docker compose -f ..\docker-compose-rag.yml down -v
docker compose -f ..\docker-compose-reranker.yml down -v

:: 사용자 정의 네트워크 삭제 (기본 네트워크 제외)
echo Docker 사용자 정의 네트워크 삭제 중...
for /f "tokens=*" %%i in ('docker network ls --filter "type=custom" -q') do (
    docker network rm %%i 2>nul || echo 네트워크 %%i는 사용 중이거나 이미 삭제되었습니다.
)

:: 추가 정리 작업
echo 1. 실행 중인 컨테이너 중지 및 삭제
for /f "tokens=*" %%i in ('docker ps -aq') do (
    docker stop %%i 2>nul
    docker rm %%i 2>nul
)

echo 2. 관련 이미지 삭제
for /f "tokens=*" %%i in ('docker images --filter "reference=uwsgi-fastcgi*" -q') do docker rmi -f %%i 2>nul
for /f "tokens=*" %%i in ('docker images --filter "reference=milvus*" -q') do docker rmi -f %%i 2>nul
for /f "tokens=*" %%i in ('docker images --filter "reference=reranker*" -q') do docker rmi -f %%i 2>nul
for /f "tokens=*" %%i in ('docker images --filter "reference=*rag*" -q') do docker rmi -f %%i 2>nul
for /f "tokens=*" %%i in ('docker images --filter "reference=*api-gateway*" -q') do docker rmi -f %%i 2>nul

echo 3. 관련 볼륨 삭제
docker volume rm shared_tmp 2>nul
for /f "tokens=*" %%i in ('docker volume ls -q --filter "name=uwsgi-fastcgi"') do docker volume rm %%i 2>nul

echo 5. 빌드 캐시 정리
docker builder prune -f

echo === 정리 완료 ===
echo RAG 시스템 관련 모든 Docker 리소스가 정리되었습니다.

:End
pause 