# RAG 시스템 테스트 명령어 모음

## 1. 전체 서비스 시작
```bash
# 모든 서비스 시작 (RAG + Reranker + Prompt + Ollama + DB)
./scripts/setup.sh all
```

## 2. 데이터 삽입 테스트
```bash
# 테스트 문서 1 삽입
curl -X POST http://localhost/rag/insert -H "Content-Type: application/json" -d '{
    "domain": "tech",
    "title": "삼성전자 반도체",
    "author": "김기자",
    "text": "삼성전자는 세계 최대의 메모리 반도체 제조업체입니다. 특히 DRAM과 NAND 플래시 분야에서 시장을 선도하고 있습니다.",
    "info": {
        "press_num": "테크뉴스",
        "url": "http://example.com/samsung"
    },
    "tags": {
        "date": "20240314",
        "category": "반도체"
    }
}'

# 테스트 문서 2 삽입
curl -X POST http://localhost/rag/insert -H "Content-Type: application/json" -d '{
    "domain": "tech",
    "title": "SK하이닉스 실적",
    "author": "이기자",
    "text": "SK하이닉스는 2024년 1분기에 메모리 반도체 시장에서 긍정적인 실적을 기록했습니다. AI 수요 증가로 인한 매출 상승이 주요 원인입니다.",
    "info": {
        "press_num": "경제일보",
        "url": "http://example.com/skhynix"
    },
    "tags": {
        "date": "20240314",
        "category": "반도체"
    }
}'

# 테스트 문서 3 삽입
curl -X POST http://localhost/rag/insert -H "Content-Type: application/json" -d '{
    "domain": "tech",
    "title": "반도체 산업 전망",
    "author": "박기자",
    "text": "글로벌 반도체 시장은 AI 붐과 함께 크게 성장할 것으로 전망됩니다. 특히 한국의 메모리 반도체 기업들이 주도적 역할을 할 것으로 예상됩니다.",
    "info": {
        "press_num": "산업신문",
        "url": "http://example.com/semiconductor"
    },
    "tags": {
        "date": "20240314",
        "category": "반도체"
    }
}'

# 테스트 문서 4 삽입
curl -X POST http://localhost/rag/insert -H "Content-Type: application/json" -d '{
    "domain": "tech",
    "title": "인텔 신규 투자",
    "author": "최기자",
    "text": "인텔이 독일에 대규모 반도체 공장 설립을 위한 투자를 진행 중입니다. 유럽의 반도체 공급망 강화를 위한 전략적 투자로 평가받고 있습니다.",
    "info": {
        "press_num": "글로벌뉴스",
        "url": "http://example.com/intel"
    },
    "tags": {
        "date": "20240314",
        "category": "반도체"
    }
}'

# 테스트 문서 5 삽입
curl -X POST http://localhost/rag/insert -H "Content-Type: application/json" -d '{
    "domain": "tech",
    "title": "TSMC 기술 혁신",
    "author": "정기자",
    "text": "TSMC가 2나노 공정 개발에 성공했다고 발표했습니다. 이는 반도체 미세공정 기술의 새로운 이정표로 평가되며, 2025년 양산을 목표로 하고 있습니다.",
    "info": {
        "press_num": "아시아경제",
        "url": "http://example.com/tsmc"
    },
    "tags": {
        "date": "20240314",
        "category": "반도체"
    }
}'
```

## 3. RAG 검색 테스트
```bash
# 직접 RAG 검색 테스트
curl -X GET "http://localhost/rag/search?query_text=한국의%20반도체%20기업들은%20어떤가요?&top_k=3"
```

## 4. Reranker 테스트
```bash
# RAG 결과를 Reranker로 재순위화
curl -X POST http://localhost/reranker/rerank -H "Content-Type: application/json" -d '{
    "query": "한국의 반도체 기업들은 어떤가요?",
    "results": [
        {
            "id": "1",
            "title": "삼성전자 반도체",
            "text": "삼성전자는 세계 최대의 메모리 반도체 제조업체입니다..."
        }
    ]
}'
```

## 5. Prompt 서비스 테스트
```bash
# 챗봇 테스트
curl -X POST http://localhost/prompt/chat -H "Content-Type: application/json" -d '{
    "query": "한국의 수도는 어디인가요?",
    "model": "gemma:2b"
}' | python3 -m json.tool

# RAG + 요약 테스트
curl -X POST http://localhost/prompt/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "query": "반도체 기업",
    "domain": "tech"
  }'
```

## 6. 서비스 상태 확인
```bash
# 각 서비스의 상태 확인
curl http://localhost/rag/
curl http://localhost/reranker/health
curl http://localhost/prompt/health
curl http://localhost:11434/api/tags  # Ollama 모델 목록
```

## 7. 서비스 종료
```bash
./scripts/shutdown.sh all
```

## 참고사항
- 현재 설정은 테스트용으로 RAG에서 3개, Reranker에서 2개의 문서만 처리하도록 되어 있습니다.
- 모든 curl 응답은 `| python3 -m json.tool` 또는 `| jq`를 붙여서 보기 좋게 포맷팅할 수 있습니다.
- 오류 발생 시 각 서비스의 로그를 확인하세요:
  ```bash
  docker logs milvus-rag
  docker logs milvus-reranker
  docker logs milvus-prompt
  docker logs milvus-ollama-cpu
  ```