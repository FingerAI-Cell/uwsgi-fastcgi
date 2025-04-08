@echo off
echo === RAG 시스템 셋업 시작 ===

echo Docker 네트워크 생성 중...
docker network create rag_network 2>nul || echo rag_network가 이미 존재합니다.

echo 통합 서비스 시작 중...
docker compose -f ..\docker-compose.yml up -d

echo 서비스 상태 확인 중...
docker ps | findstr /I "milvus api-gateway unified-nginx"

echo === 셋업 완료 ===
echo 시스템이 가동되었습니다. 다음 URL로 접근할 수 있습니다:
echo - RAG 서비스: http://localhost/
echo - Reranker 서비스: http://localhost/reranker/ (통합 서비스 내 경로)
echo - 통합 API: http://localhost/api/enhanced-search?query_text=검색어
echo - Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)
echo 참고: 개별 실행 시 Reranker 서비스는 http://localhost:8080/reranker/ 로 접근합니다.

pause 