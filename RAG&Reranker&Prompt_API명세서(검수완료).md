# RAG · Reranker · Prompt API 명세서 (최종 검수판)

> **모든 엔드포인트에 대해**
> * **기본 정보** – Method · URL · 설명
> * **요청 파라미터** – 필수 여부·타입·설명 (Query / Body 구분)
> * **요청 예시** – 완전한 `curl` 명령
> * **응답 파라미터** – 필드·타입·설명
> * **성공 응답 예시**
> * **실패 응답 예시** (가능한 경우)

---

## 목차
| # | 엔드포인트 | 설명 |
|---|------------|------|
| 1 | [/rag/](#1-rag) | RAG 상태 확인 |
| 2 | [/rag/insert](#2-raginsert) | 문서 삽입 |
| 3 | [/rag/insert/raw](#3-raginsertraw) | 문서 원본 삽입 |
| 4 | [/rag/search](#4-ragsearch) | 문서 검색 |
| 5 | [/rag/delete](#5-ragdelete) | 문서 삭제 |
| 6 | [/rag/document](#6-ragdocument) | 문서·패시지 조회 |
| 7 | [/rag/domains](#7-ragdomains) | 도메인 목록 조회 |
| 8 | [/rag/domains/delete](#8-ragdomainsdelete) | 도메인 삭제 |
| 9 | [/rag/data/show](#9-ragdatashow) | 컬렉션 정보 조회 |
| 10 | [/reranker/health](#10-rerankerhealth) | Reranker 상태 |
| 11 | [/reranker/enhanced-search](#11-rerankerenhanced-search) | 통합 검색(재랭킹) |
| 12 | [/reranker/rerank](#12-rerankerrerank) | 단건 재랭킹 (FlashRank/MRC/하이브리드) |
| 13 | [/reranker/batch_rerank](#13-rerankerbatch_rerank) | 배치 재랭킹 (FlashRank/MRC/하이브리드) |
| 14 | [/reranker/hybrid-rerank](#14-rerankerhybrid-rerank) | 하이브리드 재랭킹 전용 엔드포인트 |
| 15 | [/prompt/health](#15-prompthealth) | Prompt 상태 |
| 16 | [/prompt/enhanced_search](#16-promptenhanced_search) | RAG 검색 결과를 Reranker로 재랭킹 (LLM 요약 없음) |
| 17 | [/prompt/summarize](#17-promptsummarize) | 문서 요약 |
| 18 | [/prompt/chat](#18-promptchat) | 챗봇 응답 |
| 19 | [/prompt/models](#19-promptmodels) | 모델 목록 |
| 20 | [/vision/health](#20-visionhealth) | Vision 상태 |
| 21 | [/vision/analyze](#21-visionanalyze) | 이미지 분석 |

> **모든 URL** 는 `http://localhost` 기준이며, 실제 배포 시 호스트/포트를 맞춰 수정하세요.

---

## 1. /rag/
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **GET** |
| URL | `/rag/` |
| 설명 | RAG FastCGI 서비스 헬스 체크 |

### 요청 파라미터
없음

### 요청 예시
```bash
curl -X GET http://localhost/rag/
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| message | String | 상태 메시지 |

### 성공 응답 예시
```json
{ "message": "Hello, FastCGI is working!" }
```

---

## 2. /rag/insert
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **POST** |
| URL | `/rag/insert` |
| Content‑Type | `application/json` |
| 설명 | 문서를 Milvus 컬렉션에 삽입 |

### 요청 파라미터 (Body)
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| documents | Y | Array | 삽입할 문서 배열 |
| ignore | N | Boolean | 중복 문서 무시 여부 (기본값: true) |

각 문서 객체의 구조:
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| domain | Y | String | 컬렉션 이름 (예: `news`) |
| title | Y | String | 문서 제목 |
| author | Y | String | 작성자/기관 |
| text | Y | String | 본문 |
| info | N | Object | `{ press_num, url }` |
| tags | Y | Object | `{ date(YYYYMMDD), user }` |

### 요청 예시
```bash
curl -X POST http://localhost/rag/insert \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "domain": "news",
        "title": "메타버스 뉴스",
        "author": "삼성전자",
        "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다...",
        "info": { "press_num": "비즈니스 워치", "url": "http://example.com/news/1" },
        "tags": { "date": "20240315", "user": "admin" }
      }
    ],
    "ignore": true
  }'
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| status | String | 전체 처리 상태 ("success", "partial_success", "partial_error", "error") |
| message | String | 처리 결과 메시지 |
| status_counts | Object | 상태별 처리 건수 (`success`, `skipped`, `error`) |
| results | Array | 각 문서별 처리 결과 |

### 성공 응답 예시
```json
{
  "status": "success",
  "message": "총 3개 문서 중 2개 성공, 1개 건너뜀",
  "status_counts": {
    "success": 2,
    "skipped": 1,
    "error": 0
  },
  "results": [
    {
      "status": "success",
      "message": "문서가 성공적으로 저장되었습니다.",
      "doc_id": "1234567890abcdef...",
      "raw_doc_id": "20240315-메타버스 뉴스-삼성전자",
      "domain": "news",
      "title": "메타버스 뉴스"
    },
    {
      "status": "skipped",
      "message": "이미 존재하는 문서로 건너뛰었습니다.",
      "doc_id": "abcdef1234567890...",
      "raw_doc_id": "20240315-AI 뉴스-LG전자",
      "domain": "news",
      "title": "AI 뉴스"
    }
  ]
}
```

### 실패 응답 예시
```json
{
  "status": "error",
  "message": "요청 본문이 비어있습니다.",
  "error_code": "F000001"
}
```

---

## 3. /rag/insert/raw
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **POST** |
| URL | `/rag/insert/raw` |
| Content‑Type | `application/json` |
| 설명 | 텍스트를 분할하지 않고 그대로 저장 |

### 요청 파라미터 (Body)
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| documents | Y | Array | 삽입할 문서 배열 |
| ignore | N | Boolean | 중복 문서 처리 (기본값: true) |

각 문서 객체의 구조:
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| doc_id | Y | String | 사용자 지정 문서 ID |
| passage_id | Y | Integer | 사용자 지정 패시지 ID |
| domain | Y | String | 컬렉션 이름 |
| title | Y | String | 문서 제목 |
| author | Y | String | 작성자/기관 |
| text | Y | String | 본문 |
| info | N | Object | `{ press_num, url }` |
| tags | Y | Object | `{ date(YYYYMMDD), user }` |

### 요청 예시
```bash
curl -X POST http://localhost/rag/insert/raw \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "doc_id": "unique_document_id",
        "passage_id": 1,
        "domain": "news",
        "title": "메타버스 뉴스",
        "author": "삼성전자",
        "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다...",
        "info": { "press_num": "비즈니스 워치", "url": "http://example.com/news/1" },
        "tags": { "date": "20240315", "user": "admin" }
      }
    ],
    "ignore": true
  }'
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| status | String | 전체 처리 상태 ("success", "partial_success", "partial_error", "error") |
| message | String | 처리 결과 메시지 |
| status_counts | Object | 상태별 처리 건수 (`success`, `updated`, `skipped`, `error`) |
| results | Array | 각 문서별 처리 결과 |

### 성공 응답 예시
```json
{
  "status": "success",
  "message": "총 3개 문서 중 2개 성공, 1개 업데이트, 0개 건너뜀, 0개 실패",
  "status_counts": {
    "success": 2,
    "updated": 1,
    "skipped": 0,
    "error": 0
  },
  "results": [
    {
      "status": "success",
      "message": "문서가 성공적으로 저장되었습니다.",
      "doc_id": "unique_document_id",
      "passage_id": 1,
      "domain": "news",
      "title": "메타버스 뉴스"
    }
  ]
}
```

---

## 4. /rag/search
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **POST** |
| URL | `/rag/search` |
| Content‑Type | `application/json` |
| 설명 | Milvus에서 유사 문서 검색 |

### 요청 파라미터 (Body)
| 필드 | 필수 | Type | 기본 | 설명 |
|------|------|------|------|------|
| query_text | Y | String | – | 검색어 |
| top_k | N | Integer | 5 | 검색 결과 수 |
| domains | N | Array | [] | 도메인 필터 (복수 지정 가능, 빈 배열이거나 없는 경우 모든 도메인에서 검색) |
| author | N | String | – | 작성자 필터 |
| start_date | N | String | – | 시작일 `YYYYMMDD` |
| end_date | N | String | – | 종료일 `YYYYMMDD` |
| title | N | String | – | 제목 검색 |
| info_filter | N | Object | – | `info` 필드 필터링 |
| tags_filter | N | Object | – | `tags` 필드 필터링 |

### 요청 예시
```bash
curl -X POST http://localhost/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "메타버스",
    "top_k": 5,
    "domains": ["news", "test"],
    "start_date": "20240301",
    "end_date": "20240315"
  }'
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| result_code | String | `F000000` 성공 등 |
| message | String | 결과 메시지 |
| search_params | Object | 적용된 검색 파라미터 |
| total_results | Integer | 전체 검색 결과 수 |
| returned_results | Integer | 반환된 결과 수 |
| domain_results | Object | 도메인별 검색 결과 |

각 도메인 결과 객체의 구조:
| 필드 | Type | 설명 |
|------|------|------|
| total_hits | Integer | 해당 도메인의 전체 결과 수 |
| results | Array | 검색 결과 배열 |

각 검색 결과 객체의 구조:
| 필드 | Type | 설명 |
|------|------|------|
| doc_id | String | 문서 ID |
| raw_doc_id | String | 원본 문서 ID |
| passage_id | Integer | 패시지 ID |
| title | String | 제목 |
| author | String | 작성자 |
| text | String | 본문 내용 |
| info | Object | 추가 정보 |
| tags | Object | 태그 정보 |
| score | Float | 유사도 점수 |

### 성공 응답 예시
```json
{
  "result_code": "F000000",
  "message": "검색이 성공적으로 완료되었습니다.",
  "search_params": {
    "query_text": "메타버스",
    "domains": ["news", "test"],
    "top_k": 5,
    "filters": {
      "date_range": {
        "start": "20240301",
        "end": "20240315"
      }
    }
  },
  "total_results": 10,
  "returned_results": 5,
  "domain_results": {
    "news": {
      "total_hits": 7,
      "results": [
        {
          "doc_id": "109f405744d2f1e0eccb880c70c6c6e9...",
          "raw_doc_id": "20240315-메타버스 뉴스-삼성전자",
          "passage_id": 1,
          "title": "메타버스 뉴스",
          "author": "삼성전자",
          "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다...",
          "info": {
            "press_num": "비즈니스 워치",
            "url": "http://example.com/news/1"
          },
          "tags": {
            "date": "20240315",
            "user": "admin"
          },
          "score": 0.95
        }
      ]
    },
    "test": {
      "total_hits": 3,
      "results": [
        // ... test 도메인의 결과들
      ]
    }
  }
}
```

### 실패 응답 예시
```json
{
  "result_code": "F000001",
  "message": "검색어(query_text)는 필수 입력값입니다.",
  "search_result": null
}
```

---

## 5. /rag/delete
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **DELETE** |
| URL | `/rag/delete` |
| 설명 | 문서 삭제 |

### 요청 파라미터 (Query)
| 이름 | 필수 | Type | 설명 |
|------|------|------|------|
| doc_id | Y | String | 문서 ID (원본 raw_doc_id 또는 해시된 doc_id 모두 사용 가능) |
| domains | Y | Array | 삭제할 도메인 배열 |

> **doc_id 처리 방식**
> - 원본 문서 ID(raw_doc_id) 또는 해시된 문서 ID(doc_id) 모두 사용 가능
> - 시스템이 자동으로 ID 형식을 감지하여 처리
> - 해시된 ID의 경우: 64자 길이의 16진수 문자열

### 요청 예시
```bash
curl -X DELETE "http://localhost/rag/delete?doc_id=20240315-메타버스-뉴스&domains=news&domains=test"
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| status | String | `"received"` |

### 성공 응답 예시
```json
{ "status": "received" }
```

### 실패 응답 예시
```json
{
  "error": "domains is required",
  "message": "도메인은 필수 입력값입니다."
}
```

---

## 6. /rag/document
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **POST** |
| URL | `/rag/document` |
| Content‑Type | `application/json` |
| 설명 | 특정 문서 혹은 패시지 조회 |

### 요청 파라미터 (Body)
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| doc_id | Y | String | 문서 ID (원본 raw_doc_id 또는 해시된 doc_id 모두 사용 가능) |
| domains | Y | Array | 검색할 도메인 배열 |
| passage_id | N | Integer | 패시지 ID (특정 패시지만 조회 시) |

> **doc_id 처리 방식**
> - 원본 문서 ID(raw_doc_id) 또는 해시된 문서 ID(doc_id) 모두 사용 가능
> - 시스템이 자동으로 ID 형식을 감지하여 처리
> - 해시된 ID의 경우: 64자 길이의 16진수 문자열
> - 응답에는 항상 두 가지 ID가 모두 포함됨 (doc_id, raw_doc_id)

### 요청 예시
```bash
curl -X POST http://localhost/rag/document \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "20240315-메타버스-뉴스",
    "domains": ["news", "test"],
    "passage_id": 1
  }'
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| doc_id | String | 해시된 문서 ID |
| raw_doc_id | String | 원본 문서 ID |
| domain_results | Object | 도메인별 결과 |

도메인별 결과 객체의 구조:
| 필드 | Type | 설명 |
|------|------|------|
| doc_id | String | 해시된 문서 ID |
| raw_doc_id | String | 원본 문서 ID |
| domain | String | 도메인명 |
| title | String | 문서 제목 |
| author | String | 작성자 |
| info | Object | 추가 정보 |
| tags | Object | 태그 정보 |
| passages | Array | 패시지 목록 |

패시지 객체의 구조:
| 필드 | Type | 설명 |
|------|------|------|
| passage_id | Integer | 패시지 ID |
| text | String | 패시지 내용 |
| position | Integer | 패시지 순서 |

### 성공 응답 예시 (전체 문서 조회)
```json
{
  "doc_id": "해시된_문서_ID",
  "raw_doc_id": "원본_문서_ID",
  "domain_results": {
    "news": {
      "doc_id": "해시된_문서_ID",
      "raw_doc_id": "원본_문서_ID",
      "domain": "news",
      "title": "메타버스 뉴스",
      "author": "삼성전자",
      "info": {
        "press_num": "비즈니스 워치",
        "url": "http://example.com/news/1"
      },
      "tags": {
        "date": "20240315",
        "user": "admin"
      },
      "passages": [
        {
          "passage_id": 1,
          "text": "메타버스는...",
          "position": 1
        }
      ]
    },
    "test": {
      "doc_id": "해시된_문서_ID",
      "raw_doc_id": "원본_문서_ID",
      "domain": "test",
      "title": "메타버스 테스트",
      "author": "LG전자",
      "info": { ... },
      "tags": { ... },
      "passages": [ ... ]
    }
  }
}
```

### 성공 응답 예시 (특정 패시지 조회)
```json
{
  "doc_id": "20240315-메타버스-뉴스",
  "domains": ["news", "test"],
  "title": "메타버스 뉴스",
  "passage_id": 1,
  "text": "메타버스는...",
  "position": 1
}
```

### 실패 응답 예시 (문서 없음)
```json
{
  "error": "Document not found",
  "doc_id": "20230101-존재-하지-않음",
  "domains": ["news", "test"],
  "message": "요청하신 문서를 찾을 수 없습니다."
}
```

### 실패 응답 예시 (패시지 없음)
```json
{
  "error": "Passage not found",
  "doc_id": "20240315-메타버스-뉴스",
  "passage_id": 999,
  "domains": ["news", "test"],
  "message": "요청하신 패시지를 찾을 수 없습니다."
}
```

---

## 7. /rag/domains
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **GET** |
| URL | `/rag/domains` |
| 설명 | 시스템에 등록된 모든 도메인(컬렉션) 목록 조회 |

### 요청 파라미터
없음

### 요청 예시
```bash
curl -X GET http://localhost/rag/domains
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| result_code | String | `S000000` 성공 등 |
| message | String | 결과 메시지 |
| domains | Array | 도메인(컬렉션) 정보 배열 |

각 도메인 정보 객체의 구조:
| 필드 | Type | 설명 |
|------|------|------|
| name | String | 도메인(컬렉션) 이름 |
| entity_count | Integer | 도메인 내 엔티티 수 |
| error | String | (선택) 에러 발생 시 에러 메시지 |

### 성공 응답 예시
```json
{
  "result_code": "S000000",
  "message": "도메인 목록을 성공적으로 조회했습니다.",
  "domains": [
    {
      "name": "news",
      "entity_count": 1250
    },
    {
      "name": "test",
      "entity_count": 42
    }
  ]
}
```

### 실패 응답 예시
```json
{
  "result_code": "F000010",
  "message": "도메인 목록 조회에 실패했습니다: 연결 오류",
  "domains": []
}
```

---

## 8. /rag/domains/delete
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **POST** |
| URL | `/rag/domains/delete` |
| Content‑Type | `application/json` |
| 설명 | 지정된 도메인(컬렉션)을 모든 엔티티와 함께 완전히 삭제 |

### 요청 파라미터 (Body)
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| domains | Y | String/Array | 삭제할 도메인 이름 (단일 문자열 또는 문자열 배열) |

### 요청 예시
```bash
curl -X POST http://localhost/rag/domains/delete \
  -H "Content-Type: application/json" \
  -d '{
    "domains": ["test_domain", "old_collection"]
  }'
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| result_code | String | `S000000` 성공 등 |
| status | String | 전체 처리 상태 ("success", "error", "partial", "not_found") |
| message | String | 처리 결과 메시지 |
| results | Array | 각 도메인별 처리 결과 |

각 도메인 처리 결과의 구조:
| 필드 | Type | 설명 |
|------|------|------|
| name | String | 도메인 이름 |
| status | String | 상태 ("success", "error", "not_found") |
| entity_count | Integer/String | 삭제된 엔티티 수 (또는 "알 수 없음") |
| message | String | 처리 결과 메시지 |

### 성공 응답 예시
```json
{
  "result_code": "S000000",
  "status": "success",
  "message": "총 2개 도메인 중 2개 삭제 성공, 0개 실패, 0개 없음",
  "results": [
    {
      "name": "test_domain",
      "status": "success",
      "entity_count": 120,
      "message": "도메인이 성공적으로 삭제되었습니다."
    },
    {
      "name": "old_collection",
      "status": "success",
      "entity_count": 35,
      "message": "도메인이 성공적으로 삭제되었습니다."
    }
  ]
}
```

### 실패 응답 예시 (존재하지 않는 도메인)
```json
{
  "result_code": "F000005",
  "status": "not_found",
  "message": "총 2개 도메인 중 0개 삭제 성공, 0개 실패, 2개 없음",
  "results": [
    {
      "name": "unknown_domain",
      "status": "not_found",
      "message": "존재하지 않는 도메인입니다."
    },
    {
      "name": "missing_collection",
      "status": "not_found", 
      "message": "존재하지 않는 도메인입니다."
    }
  ]
}
```

### 실패 응답 예시 (시스템 오류)
```json
{
  "result_code": "F000010",
  "message": "도메인 삭제에 실패했습니다: 연결 오류",
  "status": "error"
}
```

---

## 9. /rag/data/show
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **GET** |
| URL | `/rag/data/show` |

### 요청 파라미터 (Query)
| 이름 | 필수 | Type | 설명 |
|------|------|------|------|
| collection_name | Y | String | 컬렉션 이름 |

### 요청 예시
```bash
curl -G http://localhost/rag/data/show --data-urlencode "collection_name=news"
```

### 응답 파라미터 (정상)
| 필드 | Type | 설명 |
|------|------|------|
| schema | Object | 컬렉션 스키마 |
| partition_names | Array | 파티션 목록 |
| partition_nums | Object | 파티션 → 엔티티 수 |

### 성공 응답 예시
```json
{
  "schema": { "fields": [ { "name": "doc_id", "type": "VARCHAR" } ] },
  "partition_names": [ "p1" ],
  "partition_nums": { "p1": 100 }
}
```

### 실패 응답 예시 (없는 컬렉션 – **HTTP 200**)
```json
{
  "error": "유효한 Collection 이름을 입력해야 합니다.",
  "collection list": ["news", "description"]
}
```

---

## 10. /reranker/health
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **GET** |
| URL | `/reranker/health` |
| 설명 | Reranker 헬스 체크 |

### 요청 파라미터
없음

### 요청 예시
```bash
curl -X GET http://localhost/reranker/health
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| status | String | "ok" |
| service | String | "reranker" |

### 성공 응답 예시
```json
{ "status": "ok", "service": "reranker" }
```

---

## 11. /reranker/enhanced-search
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **GET** |
| URL | `/reranker/enhanced-search` |
| 설명 | RAG 검색 결과를 가져와 Reranker로 재랭킹 후 반환 |

### 요청 파라미터 (Query)
| 이름 | 필수 | Type | 기본 | 설명 |
|------|------|------|------|------|
| query_text | Y | String | – | 검색어 |
| top_k | N | Integer | 5 | 최종 결과 수 |
| raw_results | N | Integer | 20 | RAG에서 가져올 초기 결과 수 |
| domain | N | String | – | 도메인 필터 |
| author | N | String | – | 작성자 필터 |
| start_date | N | String | – | 시작일 `YYYYMMDD` |
| end_date | N | String | – | 종료일 `YYYYMMDD` |
| title | N | String | – | 제목 검색 |

### 요청 예시
```bash
curl -G http://localhost/reranker/enhanced-search \
  --data-urlencode "query_text=메타버스 최신 동향" \
  --data-urlencode "top_k=5" \
  --data-urlencode "raw_results=20" \
  --data-urlencode "domain=news"
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| result_code | String | `F000000` 성공 등 |
| message | String | 결과 메시지 |
| search_params | Object | 실제 적용 파라미터 |
| search_result | Array | 재랭킹된 결과 목록 |

### 성공 응답 예시
```json
{
  "result_code": "F000000",
  "message": "검색 및 재랭킹이 성공적으로 완료되었습니다.",
  "search_params": {
    "query_text": "메타버스 최신 동향",
    "top_k": 5,
    "raw_results": 20,
    "filters": { "domain": "news" }
  },
  "search_result": [
    {
      "doc_id": "20240315-메타버스-뉴스",
      "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다...",
      "score": 0.98,
      "title": "메타버스 뉴스",
      "author": "삼성전자"
    }
  ]
}
```

### 실패 응답 예시 (검색어 누락)
```json
{
  "result_code": "F000001",
  "message": "검색어(query_text)는 필수 입력값입니다.",
  "search_result": null
}
```

---

## 12. /reranker/rerank
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **POST** |
| URL | `/reranker/rerank` |
| Content‑Type | `application/json` |
| 설명 | 단일 쿼리와 결과 목록을 재랭킹 |

### 요청 파라미터
*Query*:
| 이름 | 필수 | Type | 기본값 | 설명 |
|------|------|------|------|------|
| top_k | N | Integer | - | 반환할 최대 결과 수 |
| reranker_type | N | String | "default" | 재랭킹 방식 ("default", "hybrid", "mrc", "flashrank") |

*Body*:
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| query | Y | String | 검색 쿼리 |
| results | Y | Array | 재랭킹할 문서 배열 |
| total | N | Integer | 전체 결과 수 |
| reranked | N | Boolean | 이미 재랭킹 되었는지 여부 |
| mrc_weight | N | Float | MRC 점수 가중치 (0.0~1.0, 하이브리드 모드에서만 사용) |

각 결과 객체의 구조:
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| passage_id | N | Any | 패시지 ID |
| doc_id | N | String | 문서 ID |
| text | Y | String | 패시지 텍스트 내용 |
| score | N | Float | 원본 점수 |
| position | N | Integer | 원본 위치 |
| metadata | N | Object | 메타데이터 객체 |

metadata 객체의 구조:
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| title | N | String | 문서 제목 |
| author | N | String | 작성자 |
| tags | N | Object | 태그 정보 (예: date) |
| doc_id | N | String | 문서 ID (중복 가능) |
| original_score | N | Float | 원본 점수 |

### 요청 예시
```bash
curl -X POST "http://localhost/reranker/rerank?top_k=3&reranker_type=hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "인공지능이란 무엇인가?",
    "results": [
      {
        "id": "1",
        "text": "인공지능(AI)은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술이다.",
        "meta": {
          "doc_id": "doc001",
          "original_score": 0.85
        }
      },
      {
        "id": "2",
        "text": "머신러닝은 인공지능의 한 분야로, 컴퓨터가 데이터로부터 학습하여 패턴을 찾아내는 방법론이다.",
        "meta": {
          "doc_id": "doc002",
          "original_score": 0.75
        }
      }
    ],
    "mrc_weight": 0.7
  }'
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| query | String | 요청한 검색 쿼리 |
| results | Array | 재랭킹된 결과 배열 |
| total | Integer/null | 결과 총 개수 |
| reranked | Boolean | 재랭킹 완료 여부 (true: 재랭킹 성공) |
| reranker_type | String | "hybrid" (고정값) |
| processing_time | Float | 처리 소요 시간(초) |
| flashrank_time | Float | FlashRank 처리 시간(초) |
| mrc_time | Float | MRC 처리 시간(초) |
| mrc_weight | Float | 사용된 MRC 가중치 |

각 결과 객체의 구조:
| 필드 | Type | 설명 |
|------|------|------|
| id | String | 결과 ID |
| text | String | 패시지 텍스트 내용 |
| meta | Object | 메타데이터 객체 |
| score | Float | 최종 하이브리드 점수 |
| mrc_answer | String | MRC 모델이 추출한 답변 텍스트 |
| mrc_char_ids | Array | 답변의 시작/끝 문자 인덱스 [start, end] |
| mrc_score | Float | MRC 모델 점수 |
| flashrank_score | Float | FlashRank 점수 |
| hybrid_score | Float | 하이브리드 최종 점수 (score와 동일) |
| metadata | Object | 추가 메타데이터 (원본 점수 포함) |

### 성공 응답 예시
```json
{
  "query": "인공지능이란 무엇인가?",
  "results": [
    {
      "id": "1",
      "text": "인공지능(AI)은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술이다.",
      "meta": {
        "doc_id": "doc001",
        "original_score": 0.85
      },
      "score": 0.9650318562984466,
      "mrc_answer": "인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술",
      "mrc_char_ids": [10, 55],
      "mrc_score": 0.9698712229728699,
      "flashrank_score": 0.9537400007247925,
      "hybrid_score": 0.9650318562984466,
      "metadata": {
        "flashrank_score": 0.9537400007247925,
        "mrc_score": 0.9698712229728699
      }
    },
    {
      "id": "2",
      "text": "머신러닝은 인공지능의 한 분야로, 컴퓨터가 데이터로부터 학습하여 패턴을 찾아내는 방법론이다.",
      "meta": {
        "doc_id": "doc002",
        "original_score": 0.75
      },
      "score": 0.3672780990600586,
      "mrc_answer": "컴퓨터가 데이터로부터 학습하여 패턴을 찾아내는 방법론",
      "mrc_char_ids": [19, 48],
      "mrc_score": 0.3194928467273712,
      "flashrank_score": 0.4787770211696625,
      "hybrid_score": 0.3672780990600586,
      "metadata": {
        "flashrank_score": 0.4787770211696625,
        "mrc_score": 0.3194928467273712
      }
    }
  ],
  "total": 2,
  "reranked": true,
  "reranker_type": "hybrid",
  "processing_time": 1.0239102840423584,
  "flashrank_time": 0.0,
  "mrc_time": 0.3780794143676758,
  "mrc_weight": 0.7
}
```

### 실패 응답 예시
```json
{
  "error": "No JSON data provided"
}
```

---

## 13. /reranker/batch_rerank
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **POST** |
| URL | `/reranker/batch_rerank` |
| Content‑Type | `application/json` |
| 설명 | 여러 쿼리를 한 번에 재랭킹 |

### 요청 파라미터
*Query*:
| 이름 | 필수 | Type | 기본값 | 설명 |
|------|------|------|------|------|
| top_k | N | Integer | - | 각 쿼리별 반환할 최대 결과 수 |
| reranker_type | N | String | "default" | 재랭킹 방식 ("default", "hybrid", "mrc", "flashrank") |

*Body*:
요청 본문은 `SearchResultModel` 객체의 배열입니다. 각 객체는 다음 구조를 가집니다:

| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| query | Y | String | 검색 쿼리 |
| results | Y | Array | 재랭킹할 문서 배열 |
| total | N | Integer | 전체 결과 수 |
| reranked | N | Boolean | 이미 재랭킹 되었는지 여부 |
| mrc_weight | N | Float | MRC 점수 가중치 (0.0~1.0, 하이브리드 모드에서만 사용) |

각 결과 객체의 구조:
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| passage_id | N | Any | 패시지 ID |
| doc_id | N | String | 문서 ID |
| text | Y | String | 패시지 텍스트 내용 |
| score | N | Float | 원본 점수 |
| position | N | Integer | 원본 위치 |
| metadata | N | Object | 메타데이터 객체 |

metadata 객체의 구조:
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| title | N | String | 문서 제목 |
| author | N | String | 작성자 |
| tags | N | Object | 태그 정보 (예: date) |
| doc_id | N | String | 문서 ID (중복 가능) |
| original_score | N | Float | 원본 점수 |

### 요청 예시
```bash
curl -X POST "http://localhost/reranker/batch_rerank?top_k=5&reranker_type=hybrid" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "query": "메타버스 최신 동향",
      "results": [
        {
          "passage_id": 0,
          "doc_id": "20240315-메타버스-뉴스",
          "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다...",
          "meta": {
            "doc_id": "doc001",
            "original_score": 0.95
          }
        },
        {
          "passage_id": 1,
          "doc_id": "20240310-가상현실-동향",
          "text": "가상현실 기술은 게임뿐 아니라 교육, 의료 등 다양한 분야로 확장되고 있다...",
          "meta": {
            "doc_id": "doc002",
            "original_score": 0.85
          }
        }
      ],
      "mrc_weight": 0.7
    },
    {
      "query": "가상현실 시장 전망",
      "results": [
        {
          "passage_id": 0,
          "doc_id": "20240312-VR-뉴스",
          "text": "가상현실 시장은 2024년 급격한 성장이 예상된다...",
          "meta": {
            "doc_id": "doc003",
            "original_score": 0.92
          }
        }
      ],
      "mrc_weight": 0.7
    }
  ]'
```

### 응답 파라미터
응답은 각 쿼리에 대한 재랭킹 결과 배열입니다. 각 항목은 다음 구조를 가집니다:

| 필드 | Type | 설명 |
|------|------|------|
| query | String | 요청한 검색 쿼리 |
| results | Array | 재랭킹된 결과 배열 |
| total | Integer/null | 결과 총 개수 |
| reranked | Boolean | 재랭킹 완료 여부 (true: 재랭킹 성공, false: 재랭킹 실패/미수행) |
| reranker_type | String | 사용된 재랭킹 방식 ("default", "hybrid", "mrc", "flashrank") |
| processing_time | Float | 처리 소요 시간(초) |

각 결과 객체의 구조:
| 필드 | Type | 설명 |
|------|------|------|
| id | String | 결과 ID |
| text | String | 패시지 텍스트 내용 |
| meta | Object | 메타데이터 객체 |
| score | Float | 최종 재랭킹 점수 (높을수록 관련성 높음) |
| mrc_answer | String | MRC 모델이 추출한 답변 텍스트 (MRC/하이브리드 모드에서만) |
| mrc_char_ids | Array | 답변의 시작/끝 문자 인덱스 [start, end] (MRC/하이브리드 모드에서만) |
| mrc_score | Float | MRC 모델 점수 (MRC/하이브리드 모드에서만) |
| flashrank_score | Float | FlashRank 점수 (FlashRank/하이브리드 모드에서만) |
| hybrid_score | Float | 하이브리드 최종 점수 (하이브리드 모드에서만) |
| metadata | Object | 추가 메타데이터 (원본 점수 포함) |

### 성공 응답 예시
```json
[
  {
    "query": "메타버스 최신 동향",
    "results": [
      {
        "id": "1",
        "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다...",
        "meta": {
          "doc_id": "doc001",
          "original_score": 0.95
        },
        "score": 0.9650318562984466,
        "mrc_answer": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다...",
        "mrc_char_ids": [10, 55],
        "mrc_score": 0.9698712229728699,
        "flashrank_score": 0.9537400007247925,
        "hybrid_score": 0.9650318562984466,
        "metadata": {
          "flashrank_score": 0.9537400007247925,
          "mrc_score": 0.9698712229728699
        }
      },
      {
        "id": "2",
        "text": "가상현실 기술은 게임뿐 아니라 교육, 의료 등 다양한 분야로 확장되고 있다...",
        "meta": {
          "doc_id": "doc002",
          "original_score": 0.85
        },
        "score": 0.85,
        "mrc_answer": "가상현실 기술은 게임뿐 아니라 교육, 의료 등 다양한 분야로 확장되고 있다...",
        "mrc_char_ids": [19, 48],
        "mrc_score": 0.3194928467273712,
        "flashrank_score": 0.4787770211696625,
        "hybrid_score": 0.3672780990600586,
        "metadata": {
          "flashrank_score": 0.4787770211696625,
          "mrc_score": 0.3194928467273712
        }
      }
    ],
    "total": 2,
    "reranked": true,
    "reranker_type": "hybrid",
    "processing_time": 1.0239102840423584,
    "flashrank_time": 0.0,
    "mrc_time": 0.3780794143676758,
    "mrc_weight": 0.7
  },
  {
    "query": "가상현실 시장 전망",
    "results": [
      {
        "id": "0",
        "text": "가상현실 시장은 2024년 급격한 성장이 예상된다...",
        "meta": {
          "doc_id": "doc003",
          "original_score": 0.92
        },
        "score": 0.92,
        "mrc_answer": "가상현실 시장은 2024년 급격한 성장이 예상된다...",
        "mrc_char_ids": [10, 55],
        "mrc_score": 0.92,
        "flashrank_score": 0.92,
        "hybrid_score": 0.92,
        "metadata": {
          "flashrank_score": 0.92,
          "mrc_score": 0.92
        }
      }
    ],
    "total": 1,
    "reranked": true,
    "reranker_type": "hybrid",
    "processing_time": 0.098,
    "flashrank_time": 0.0,
    "mrc_time": 0.3780794143676758,
    "mrc_weight": 0.7
  }
]
```

### 실패 응답 예시
```json
{
  "error": "Batch reranking failed: Invalid input format"
}
```

---

## 14. /reranker/hybrid-rerank
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **POST** |
| URL | `/reranker/hybrid-rerank` |
| Content‑Type | `application/json` |
| 설명 | MRC와 FlashRank를 결합한 하이브리드 재랭킹 전용 엔드포인트 |

### 요청 파라미터
*Query*:
| 이름 | 필수 | Type | 기본값 | 설명 |
|------|------|------|------|------|
| top_k | N | Integer | - | 반환할 최대 결과 수 |

*Body*:
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| query | Y | String | 검색 쿼리 |
| results | Y | Array | 재랭킹할 문서 배열 |
| total | N | Integer | 전체 결과 수 |
| mrc_weight | N | Float | MRC 점수 가중치 (0.0~1.0, 기본값: 0.7) |

각 결과 객체의 구조:
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| id | Y | String | 결과 ID |
| text | Y | String | 패시지 텍스트 내용 |
| meta | N | Object | 메타데이터 객체 |
| score | N | Float | 원본 점수 |

meta 객체의 구조:
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| doc_id | N | String | 문서 ID |
| original_score | N | Float | 원본 점수 |

### 요청 예시
```bash
curl -X POST "http://localhost/reranker/hybrid-rerank?top_k=3" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "인공지능이란 무엇인가?",
    "results": [
      {
        "id": "1",
        "text": "인공지능(AI)은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술이다.",
        "meta": {
          "doc_id": "doc001",
          "original_score": 0.85
        }
      },
      {
        "id": "2",
        "text": "머신러닝은 인공지능의 한 분야로, 컴퓨터가 데이터로부터 학습하여 패턴을 찾아내는 방법론이다.",
        "meta": {
          "doc_id": "doc002",
          "original_score": 0.75
        }
      }
    ],
    "mrc_weight": 0.7
  }'
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| query | String | 요청한 검색 쿼리 |
| results | Array | 재랭킹된 결과 배열 |
| total | Integer/null | 결과 총 개수 |
| reranked | Boolean | 재랭킹 완료 여부 (true: 재랭킹 성공) |
| reranker_type | String | "hybrid" (고정값) |
| processing_time | Float | 처리 소요 시간(초) |
| flashrank_time | Float | FlashRank 처리 시간(초) |
| mrc_time | Float | MRC 처리 시간(초) |
| mrc_weight | Float | 사용된 MRC 가중치 |

각 결과 객체의 구조:
| 필드 | Type | 설명 |
|------|------|------|
| id | String | 결과 ID |
| text | String | 패시지 텍스트 내용 |
| meta | Object | 메타데이터 객체 |
| score | Float | 최종 하이브리드 점수 |
| mrc_answer | String | MRC 모델이 추출한 답변 텍스트 |
| mrc_char_ids | Array | 답변의 시작/끝 문자 인덱스 [start, end] |
| mrc_score | Float | MRC 모델 점수 |
| flashrank_score | Float | FlashRank 점수 |
| hybrid_score | Float | 하이브리드 최종 점수 (score와 동일) |
| metadata | Object | 추가 메타데이터 (원본 점수 포함) |

### 성공 응답 예시
```json
{
  "query": "인공지능이란 무엇인가?",
  "results": [
    {
      "id": "1",
      "text": "인공지능(AI)은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술이다.",
      "meta": {
        "doc_id": "doc001",
        "original_score": 0.85
      },
      "score": 0.9650318562984466,
      "mrc_answer": "인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술",
      "mrc_char_ids": [10, 55],
      "mrc_score": 0.9698712229728699,
      "flashrank_score": 0.9537400007247925,
      "hybrid_score": 0.9650318562984466,
      "metadata": {
        "flashrank_score": 0.9537400007247925,
        "mrc_score": 0.9698712229728699
      }
    },
    {
      "id": "2",
      "text": "머신러닝은 인공지능의 한 분야로, 컴퓨터가 데이터로부터 학습하여 패턴을 찾아내는 방법론이다.",
      "meta": {
        "doc_id": "doc002",
        "original_score": 0.75
      },
      "score": 0.3672780990600586,
      "mrc_answer": "컴퓨터가 데이터로부터 학습하여 패턴을 찾아내는 방법론",
      "mrc_char_ids": [19, 48],
      "mrc_score": 0.3194928467273712,
      "flashrank_score": 0.4787770211696625,
      "hybrid_score": 0.3672780990600586,
      "metadata": {
        "flashrank_score": 0.4787770211696625,
        "mrc_score": 0.3194928467273712
      }
    }
  ],
  "total": 2,
  "reranked": true,
  "reranker_type": "hybrid",
  "processing_time": 1.0239102840423584,
  "flashrank_time": 0.0,
  "mrc_time": 0.3780794143676758,
  "mrc_weight": 0.7
}
```

### 실패 응답 예시
```json
{
  "error": "No JSON data provided"
}
```

---

## 15. /prompt/health
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **GET** |
| URL | `/prompt/health` |
| 설명 | Prompt‑Backend 헬스 체크 |

### 요청 파라미터
없음

### 요청 예시
```bash
curl -X GET http://localhost/prompt/health
```

### 성공 응답 예시
```json
{
  "status": "ok",
  "timestamp": "2025-04-22T12:34:56Z",
  "service": "prompt-backend"
}
```

---

## 16. /prompt/enhanced_search
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **POST** |
| URL | `/prompt/enhanced_search` |
| Content‑Type | `application/json` |
| 설명 | RAG 검색 결과를 Reranker로 재랭킹 (LLM 요약 없음) |

### 요청 파라미터 (Body)
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| query | Y | String | 검색 쿼리 |
| top_m | N | Integer | RAG 검색 결과 최대 수 (기본값: config의 search_top) |
| top_n | N | Integer | Reranker 결과 최대 수 (기본값: config의 rerank_top) |
| threshold | N | Float | 재랭킹 점수 임계치 (기본값: 0.1) - 이 값보다 낮은 점수의 결과는 필터링됨 |
| domain | N | String | 단일 도메인 필터 (지정하지 않으면 모든 도메인에서 검색) |
| domains | N | Array | 복수 도메인 필터 (빈 배열이거나 없는 경우 모든 도메인에서 검색) |
| author | N | String | 작성자 필터 |
| start_date | N | String | 시작일 `YYYYMMDD` |
| end_date | N | String | 종료일 `YYYYMMDD` |
| title | N | String | 제목 검색 |
| info_filter | N | Object | `info` 필드 필터링 |
| tags_filter | N | Object | `tags` 필드 필터링 |

### 요청 예시
```bash
curl -X POST http://localhost/prompt/enhanced_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "메타버스 최신 동향",
    "top_m": 50,
    "top_n": 10,
    "threshold": 0.2,
    "domain": "news",
    "start_date": "20240301",
    "end_date": "20240331"
  }'
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| query | String | 요청한 쿼리 |
| top_m | Integer | 요청한 RAG 검색 결과 최대 수 |
| top_n | Integer | 요청한 Reranker 결과 최대 수 |
| threshold | Float | 적용된 임계치 값 |
| search_count | Integer | 실제 검색된 문서 수 |
| reranked_count | Integer | 실제 재랭킹된 문서 수 |
| filtered_count | Integer | 임계치 필터링 후 남은 문서 수 |
| results | Array | 재랭킹된 결과 배열 |
| processing_time | Float | 처리 소요 시간(초) |

각 결과 항목의 구조:
| 필드 | Type | 설명 |
|------|------|------|
| passage_id | Any | 패시지 ID |
| doc_id | String | 문서 ID |
| text | String | 패시지 텍스트 내용 |
| score | Float | 점수 |
| rerank_score | Float | 재랭킹 점수 |
| rerank_position | Integer | 재랭킹 후 위치 |
| title | String | 문서 제목 |
| author | String | 작성자 |
| domain | String | 문서 도메인 |
| tags | Object | 태그 정보 (date 등) |
| info | Object | 추가 정보 (press_num, url 등) |

### 성공 응답 예시
```json
{
  "query": "메타버스 최신 동향",
  "top_m": 50,
  "top_n": 10,
  "threshold": 0.2,
  "search_count": 35,
  "reranked_count": 10,
  "filtered_count": 8,
  "results": [
    {
      "passage_id": 1,
      "doc_id": "20240310-가상현실-동향",
      "text": "가상현실 기술은 게임뿐 아니라 교육, 의료 등 다양한 분야로 확장되고 있다...",
      "score": 0.98,
      "rerank_score": 0.98,
      "rerank_position": 0,
      "title": "가상현실 동향",
      "author": "LG전자",
      "domain": "news",
      "tags": { 
        "date": "20240310" 
      },
      "info": {
        "press_num": "IT뉴스",
        "url": "http://example.com/vr-news"
      }
    },
    {
      "passage_id": 0,
      "doc_id": "20240315-메타버스-뉴스",
      "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다...",
      "score": 0.95,
      "rerank_score": 0.95,
      "rerank_position": 1,
      "title": "메타버스 뉴스",
      "author": "삼성전자",
      "domain": "tech",
      "tags": {
        "date": "20240315"
      },
      "info": {
        "press_num": "비즈니스 워치",
        "url": "http://example.com/metaverse-news"
      }
    }
  ],
  "processing_time": 0.153
}
```

### 실패 응답 예시
```json
{
  "error": "쿼리가 필요합니다"
}
```

---

## 17. /prompt/summarize
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **POST** |
| URL | `/prompt/summarize` |
| Content‑Type | `application/json` |
| 설명 | 검색 → 재랭킹 → LLM 요약 |

### 요청 파라미터 (Body)
| 필드 | 필수 | Type | 설명 |
|------|------|------|------|
| query | Y | String | 요약할 쿼리 |
| domain | N | String | 단일 도메인 필터 (지정하지 않으면 모든 도메인에서 검색) |
| domains | N | Array | 복수 도메인 필터 (빈 배열이거나 없는 경우 모든 도메인에서 검색) |
| author | N | String | 작성자 필터 |
| start_date | N | String | 시작일 `YYYYMMDD` |
| end_date | N | String | 종료일 `YYYYMMDD` |
| title | N | String | 제목 검색 |
| info_filter | N | Object | `info` 필드 필터링 |
| tags_filter | N | Object | `tags` 필드 필터링 |

### 요청 예시
```bash
curl -X POST http://localhost/prompt/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "query": "메타버스 최신 동향",
    "domain": "news",
    "start_date": "20240301",
    "end_date": "20240331"
  }'
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| query | String | 요청한 쿼리 |
| summary | String | 요약 결과 |
| documents_count | Integer | 처리된 문서 수 |
| prompt_length | Integer | 프롬프트 길이 |

### 성공 응답 예시
```json
{
  "query": "메타버스 최신 동향",
  "summary": "최근 메타버스 시장은 급성장 중...",
  "documents_count": 5,
  "prompt_length": 1024
}
```

### 실패 응답 예시
```json
{
  "error": "쿼리가 필요합니다"
}
```

---

## 18. /prompt/chat
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **POST** |
| URL | `/prompt/chat` |
| Content‑Type | `application/json` |
| 설명 | 간단한 Q&A 챗봇 |

### 요청 파라미터 (Body)
| 이름 | 필수 | Type | 설명 |
|------|------|------|------|
| query | Y | String | 질문 내용 |
| model | N | String | 사용할 모델 (기본값: 서버 설정) |
| stream | N | Boolean | 스트리밍 모드 활성화 여부 (기본값: false) |

### 요청 예시 (일반 모드)
```bash
curl -X POST http://localhost/prompt/chat \
  -H "Content-Type: application/json" \
  -d '{ "query": "메타버스란 무엇인가요?", "model": "llama2", "stream": false }'
```

### 요청 예시 (스트리밍 모드)
```bash
curl -X POST http://localhost/prompt/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{ "query": "메타버스란 무엇인가요?", "model": "llama2", "stream": true }'
```

### 성공 응답 예시 (일반 모드)
```json
{
  "query": "메타버스란 무엇인가요?",
  "model": "llama2",
  "response": "메타버스는 가상과 현실이 융합된 디지털 공간입니다..."
}
```

### 스트리밍 모드 응답
스트리밍 모드 활성화(`stream=true`) 시:
- 응답은 `text/event-stream` 형식으로 반환됩니다
- 응답은 생성되는 텍스트 조각들이 실시간으로 전송됩니다
- 클라이언트는 이를 순차적으로 수신하여 사용자에게 표시할 수 있습니다

### 실패 응답 예시
```json
{ "error": "질문이 필요합니다" }
```

---

## 19. /prompt/models
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **GET** |
| URL | `/prompt/models` |
| 설명 | 사용 가능한 LLM 모델 목록 |

### 요청 파라미터
없음

### 요청 예시
```bash
curl -X GET http://localhost/prompt/models
```

### 성공 응답 예시
```json
{
  "models": ["llama2", "mistral", "gemma"],
  "default_model": "llama2",
  "total": 3
}
```

### 실패 응답 예시
```json
{ "error": "모델 목록을 가져오는 중 오류가 발생했습니다" }
```

---

## 20. /vision/health
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **GET** |
| URL | `/vision/health` |
| 설명 | Vision 서비스 상태 확인 |

### 요청 파라미터
없음

### 요청 예시
```bash
curl -X GET http://localhost/vision/health
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| status | String | 서비스 상태 |
| service | String | "vision" |
| default_model | String | 기본 사용 모델 |

### 성공 응답 예시
```json
{
  "status": "healthy",
  "service": "vision",
  "default_model": "llama:3.2-11b-vision"
}
```

---

## 21. /vision/analyze
### 기본 정보
| 항목 | 내용 |
|------|------|
| Method | **POST** |
| URL | `/vision/analyze` |
| Content‑Type | `application/json` |
| 설명 | 이미지 분석 수행 |

### 요청 파라미터 (Body)
| 이름 | 필수 | Type | 설명 |
|------|------|------|------|
| url | Y | String | 분석할 이미지 URL |
| prompt | N | String | 분석 프롬프트 (기본값: "이 이미지에 대해 설명해주세요") |
| model | N | String | 사용할 모델 (기본값: llama:3.2-11b-vision) |

### 요청 예시
```bash
curl -X POST http://localhost/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/image.jpg",
    "prompt": "이 이미지에 대해 설명해주세요",
    "model": "llama:3.2-11b-vision"
  }'
```

### 응답 파라미터
| 필드 | Type | 설명 |
|------|------|------|
| description | String | 이미지 분석 결과 |
| image_url | String | 분석된 이미지 URL |
| model | String | 사용된 모델 |
| total_duration | Number | 총 처리 시간 (ms) |
| load_duration | Number | 모델 로딩 시간 (ms) |
| prompt_eval_count | Number | 프롬프트 평가 횟수 |
| eval_count | Number | 총 평가 횟수 |
| eval_duration | Number | 평가 소요 시간 (ms) |

### 성공 응답 예시
```json
{
  "description": "이 이미지는 푸른 하늘을 배경으로 한 현대적인 도시 풍경을 보여줍니다...",
  "image_url": "https://example.com/image.jpg",
  "model": "llama:3.2-11b-vision",
  "total_duration": 2345,
  "load_duration": 123,
  "prompt_eval_count": 50,
  "eval_count": 100,
  "eval_duration": 2000
}
```

### 실패 응답 예시
```json
{
  "error": "이미지 분석에 실패했습니다"
}
```

---

> **문서 업데이트 완료** – 필요 시 추가 요청 주세요.



