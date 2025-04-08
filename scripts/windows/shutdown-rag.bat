@echo off
echo === RAG 서비스 종료 시작 ===

echo RAG 서비스 중지 중...
docker compose -f ..\docker-compose-rag.yml down -v

echo === 종료 완료 ===
echo RAG 서비스가 중지되었습니다.
echo 더 완전한 정리를 원하시면 cleanup.bat 스크립트를 실행하세요.

pause 