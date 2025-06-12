# Reranker

검색 결과의 관련성을 개선하기 위한 재순위 기능을 제공합니다.

## 개요

Reranker는 초기 검색 결과를 쿼리와의 관련성에 따라 재정렬하여 검색 품질을 향상시키는 모듈입니다. 주요 기능:

- FlashRank 기반 빠른 재순위 처리 (기본값)
- Cross-encoder 기반 재순위 처리
- Cohere API 기반 재순위 처리 지원
- FastAPI를 이용한 REST API 제공
- 단일 쿼리 및 배치 처리 지원

## 설치

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

## 모델 다운로드

이 서비스는 기본적으로 FlashRank 라이브러리를 사용하여 재순위 처리를 합니다. 모델은 Docker 빌드 과정에서 자동으로 다운로드되지만, 수동으로도 다운로드할 수 있습니다:

```bash
# 설정에 지정된 모델 다운로드
python download_models.py

# 특정 모델 다운로드
python download_models.py --model intfloat/e5-base-v2 --dir ./models
```

### MRC 모델 파일 설치 (하이브리드 재랭킹)

하이브리드 재랭킹 기능을 사용하려면 MRC 모델 파일이 필요합니다:

1. 자동 다운로드: 서비스 시작 시 파일이 없으면 Google Drive에서 자동으로 다운로드합니다.

2. 수동 설치 방법: 자동 다운로드가 실패하거나 온프레미스 환경에서는 다음 경로에 수동으로 파일을 설치할 수 있습니다:
   - MRC 설정 파일: `reranker/models/mrc/config.json` (Google Drive ID: `1JSuDygqET5Tg7wFWsqUy2Lgfa9yoTRm0`)
   - MRC 모델 체크포인트: `reranker/models/mrc/model.ckpt` (Google Drive ID: `1KEFpxaBYrp7q8uck4r_m8pJspZGMQodl`)

3. 서버 배포 시: config.json의 "mrc" 섹션에서 파일 경로가 올바른지 확인하세요. 파일이 없으면 하이브리드 재랭킹 기능이 비활성화됩니다.

## 사용법

### 환경 변수 설정

```bash
# Cohere API 키 설정 (선택 사항, Cohere 재순위기를 사용할 경우)
export COHERE_API_KEY=your_api_key_here

# 설정 파일 경로 (선택 사항)
export RERANKER_CONFIG=/path/to/config.json
```

### API 서버 실행

```bash
cd reranker
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Docker 환경에서는 uWSGI로 실행됩니다:

```bash
# Docker Compose로 실행
docker-compose up -d
```

### API 사용 예시

#### 검색 결과 재순위 처리

```bash
curl -X 'POST' \
  'http://localhost/reranker/rerank?top_k=5' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "메타버스란 무엇인가?",
    "results": [
      {
        "passage_id": 1,
        "doc_id": "doc123",
        "text": "메타버스는 가상 세계를 의미합니다.",
        "score": 0.85
      },
      {
        "passage_id": 2,
        "doc_id": "doc456",
        "text": "메타버스 기술의 발전은 VR 기기의 보급과 함께 가속화되고 있습니다.",
        "score": 0.78
      }
    ]
  }'
```

#### 건강 상태 확인

```bash
curl -X 'GET' 'http://localhost/reranker/health'
```

## 설정

`config.json` 파일을 통해 설정 가능:

```json
{
    "reranker_type": "flashrank",
    "model_name": "intfloat/e5-base-v2",
    "model_dir": "/reranker/models",
    "device": null,
    "max_length": 512,
    "batch_size": 32,
    "api_key": null
}
```

### 설정 옵션

- `reranker_type`: 재순위 기법 ("flashrank", "cross-encoder" 또는 "cohere")
- `model_name`: 사용할 모델 이름 (flashrank 타입일 경우 "intfloat/e5-base-v2" 권장)
- `model_dir`: 로컬 모델 저장 디렉토리 
- `device`: 추론 장치 (null=자동감지, "cuda", "cpu")
- `max_length`: 최대 토큰 길이
- `batch_size`: 배치 처리 크기
- `api_key`: Cohere API 키 (환경 변수로도 설정 가능)

## 코드로 직접 사용하기

```python
from reranker.service import RerankerService

# 서비스 초기화
reranker = RerankerService("/path/to/config.json")

# 문서 재순위 처리
query = "메타버스란 무엇인가?"
passages = [
    {"text": "메타버스는 가상 세계를 의미합니다."},
    {"text": "메타버스 기술의 발전은 VR 기기의 보급과 함께 가속화되고 있습니다."}
]

reranked_passages = reranker.rerank_passages(query, passages, top_k=5)
```

## 통합 예제 (RAG 시스템과 연동)

`integration.py` 파일에서 RAG 시스템과 Reranker를 통합하는 예제를 제공합니다:

```bash
# 통합 예제 실행
python integration.py
```

이 예제는 다음과 같은 기능을 보여줍니다:

1. RAG 시스템에서 검색 결과 가져오기
2. 검색 결과를 Reranker로 재순위화
3. 문서 ID로 패시지를 가져와서 재순위화

## 성능 및 벤치마크

FlashRank를 사용하면 기존 Cross-Encoder 방식보다 크게 향상된 처리 속도를 얻을 수 있습니다:

- 처리 속도: 기존 방식 대비 최대 10-100배 빠름
- 배치 처리: 대용량 패시지 처리에 최적화
- 메모리 사용량: 효율적인 메모리 관리로 대규모 쿼리 처리 가능

일반적으로 50-100개의 패시지를 재순위화하여 상위 5-20개를 선택하는 작업에 최적화되어 있습니다. 