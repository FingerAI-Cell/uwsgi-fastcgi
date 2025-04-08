@echo off
echo === Reranker 서비스 셋업 시작 ===

echo Docker 네트워크 생성 중...
docker network create rag_network 2>nul || echo rag_network가 이미 존재합니다.

echo Reranker 서비스 시작 중...
docker compose -f ..\docker-compose-reranker.yml up -d

echo 서비스 상태 확인 중...
docker ps | findstr /I "reranker nginx"

echo === 셋업 완료 ===
echo Reranker 서비스가 가동되었습니다. 다음 URL로 접근할 수 있습니다:
echo - Reranker 서비스: http://localhost:8080/reranker/

pause 