# uwsgi-fastcgi
FastCGI 기반의 uWSGI와 Nginx를 사용하여 Docker Compose로 구성된 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 사전 준비사항
#### 1. Docker 및 Docker Compose 설치
- Windows: Docker Desktop 설치
- macOS: Docker Desktop 설치
- Linux: Docker Engine 및 Docker Compose 설치

## 프로젝트 구조
```
(루트 디렉토리)
├── nginx/                    - Nginx 설정 파일
│   ├── conf.d/               - Nginx 서버 설정
│   │   └── server_base.conf  - 기본 서버 설정
│   ├── locations-enabled/    - 활성화된 location 설정 (빈 디렉토리)
│   ├── templates/            - location 템플릿 파일
│   │   ├── rag.conf.template - RAG 서비스 템플릿
│   │   └── reranker.conf.template - Reranker 서비스 템플릿
│   └── nginx.conf           - Nginx 기본 설정
├── scripts/                  - 스크립트 디렉토리
│   ├── setup.sh              - 서비스 시작 스크립트
│   ├── cleanup.sh            - 도커 리소스 정리 스크립트
│   └── shutdown.sh           - 서비스 종료 스크립트
├── volumes/                  - 영구 데이터 저장 디렉토리
│   ├── etcd/                 - Etcd 데이터
│   ├── minio/                - MinIO 데이터
│   ├── milvus/               - Milvus 데이터
│   └── logs/                 - 로그 디렉토리
├── rag/                      - RAG 서비스 디렉토리
├── reranker/                 - Reranker 서비스 디렉토리
├── milvus/                   - Milvus 설정 디렉토리
├── flashrank/                - Flashrank 라이브러리
├── docker-compose.yml        - Docker Compose 설정
└── .env                      - 환경 변수 설정
```

## 사용 방법

### 1. 자동화 스크립트 사용

시스템 셋업과 실행을 자동화하는 스크립트를 제공합니다.

#### Linux/macOS:
```bash
# 모든 서비스 시작 (RAG + Reranker + DB)
$ ./scripts/setup.sh full

# RAG 서비스만 시작 (DB 포함)
$ ./scripts/setup.sh rag

# Reranker 서비스만 시작
$ ./scripts/setup.sh reranker

# 데이터베이스 서비스만 시작 (Milvus, Etcd, MinIO)
$ ./scripts/setup.sh db

# 애플리케이션 서비스만 시작 (RAG, Reranker, Nginx)
# DB가 이미 실행 중일 때 코드 변경 후 사용
$ ./scripts/setup.sh app-only

# 서비스 종료
$ docker compose down

# 도커 리소스 완전 정리 (모든 컨테이너, 관련 이미지, 볼륨, 네트워크 삭제)
$ ./scripts/cleanup.sh
```

### 2. 서비스 상태 확인
```bash
# 실행 중인 컨테이너 확인
$ docker ps

# 특정 서비스의 로그 확인
$ docker logs milvus-rag
$ docker logs milvus-reranker
$ docker logs milvus-nginx
```

### 3. 서비스 접근

#### 서비스 엔드포인트
- RAG 서비스: **http://localhost/rag/**
- Reranker 서비스: **http://localhost/reranker/**
- Milvus UI: **http://localhost:9001** (사용자: minioadmin, 비밀번호: minioadmin)

### 4. Nginx 설정

#### 설정 파일 구조
- 서버 설정: `nginx/conf.d/server_base.conf`
- Location 템플릿: `nginx/templates/`
  - RAG 서비스: `nginx/templates/rag.conf.template`
  - Reranker 서비스: `nginx/templates/reranker.conf.template`
- 활성화된 설정: `nginx/locations-enabled/` (setup.sh에 의해 생성됨)

서비스별로 설정을 수정해야 할 경우:
```bash
# Nginx 컨테이너 접속
$ docker exec -it milvus-nginx /bin/bash

# 설정 파일 편집
$ apt-get update -y
$ apt install vim -y
$ cd /etc/nginx/conf.d
$ vim server_base.conf
```

Nginx 수동 재시작:
```bash
# Nginx 재시작
$ docker restart milvus-nginx
```

### 5. 서비스 동작 확인
```bash
# RAG 서비스 상태 확인
$ curl http://localhost/rag/health

# Reranker 서비스 상태 확인
$ curl http://localhost/reranker/health
```

### 6. API 호출 예시

#### RAG 서비스 API (/rag/ 경로)
```bash
# 기본 검색 API (GET)
curl -X GET "http://localhost/rag/search?query_text=인공지능&top_k=5&domain=news"

# 필터링을 포함한 검색 API
curl -X GET "http://localhost/rag/search?query_text=메타버스&top_k=5&domain=tech&author=삼성전자&start_date=20230101&end_date=20231231"

# 문서 삽입 API (POST)
curl -X POST http://localhost/rag/insert \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "news",
    "title": "메타버스 뉴스",
    "author": "삼성전자",
    "text": "메타버스는 비대면 시대 뜨거운 화두로 떠올랐다...",
    "info": {
      "press_num": "비즈니스 워치",
      "url": "http://example.com/news"
    },
    "tags": {
      "date": "20220804",
      "user": "user01"
    }
  }'

# 문서 삭제 API (DELETE)
curl -X DELETE "http://localhost/rag/delete?date=20220804&title=메타버스%20뉴스&author=삼성전자&domain=news"

# 문서 조회 API (GET)
curl -X GET "http://localhost/rag/document?doc_id=20220804-메타버스%20뉴스-삼성전자"

# 컬렉션 정보 조회 API (GET)
curl -X GET "http://localhost/rag/data/show?collection_name=news"
```

#### Reranker 서비스 API (/reranker/ 경로)
```bash
# Reranker API
curl -X POST http://localhost/reranker/rerank?top_k=5 \
  -H "Content-Type: application/json" \
  -d '{
    "query": "인공지능 최신 기술",
    "results": [
      {"text": "인공지능 기술의 최신 발전 동향에 관한 문서입니다."},
      {"text": "머신러닝 알고리즘의 최근 연구 결과에 대한 분석입니다."},
      {"text": "인공지능이 다양한 산업에 미치는 영향에 대한 연구입니다."}
    ]
  }'

# Batch Reranking API
curl -X POST http://localhost/reranker/batch_rerank?top_k=3 \
  -H "Content-Type: application/json" \
  -d '[
    {
      "query": "인공지능 기술",
      "results": [
        {"text": "인공지능 관련 문서 1"},
        {"text": "인공지능 관련 문서 2"}
      ]
    },
    {
      "query": "메타버스 발전",
      "results": [
        {"text": "메타버스 관련 문서 1"},
        {"text": "메타버스 관련 문서 2"}
      ]
    }
  ]'

# 상태 확인 API
curl -X GET http://localhost/reranker/health
```

### 7. 시스템 아키텍처

#### 서비스 구성
- **Milvus 백엔드**: 벡터 검색을 위한 데이터베이스 (etcd, minio, standalone)
- **RAG 서비스**: 검색 요청을 처리하고 Milvus에서 데이터를 검색
- **Reranker 서비스**: 검색 결과를 재랭킹하여 관련성이 높은 순서로 정렬
- **Nginx**: 모든 서비스에 대한 접근을 관리하는 웹 서버

#### FastCGI 구성
- RAG와 Reranker 서비스는 uWSGI를 통해 FastCGI 프로토콜로 실행
- Nginx는 Unix 소켓을 통해 uWSGI와 통신
- 각 서비스는 독립적인 Docker 컨테이너에서 실행되지만 소켓을 공유

### 8. 개발 워크플로우

#### 1. 초기 셋업
```bash
# 모든 서비스 시작 (처음 실행 시)
$ ./scripts/setup.sh full
```

#### 2. 코드 변경 후 애플리케이션 재시작
```bash
# DB는 그대로 두고 애플리케이션만 재시작
$ ./scripts/setup.sh app-only
```

#### 3. 데이터베이스만 실행
```bash
# 데이터베이스 서비스만 시작 (개발 시작 시)
$ ./scripts/setup.sh db

# 이후 애플리케이션 변경 및 실행
$ ./scripts/setup.sh app-only
```

### 9. 볼륨 및 데이터 관리

데이터는 다음 경로에 영구적으로 저장됩니다:
- 볼륨 디렉토리: `./volumes/`
  - Etcd 데이터: `./volumes/etcd/`
  - MinIO 데이터: `./volumes/minio/`
  - Milvus 데이터: `./volumes/milvus/`
  - 로그 데이터: `./volumes/logs/`

### 10. 문제 해결

#### 일반적인 문제
- **502 Bad Gateway**: FastCGI 소켓 연결 문제. 소켓 파일이 올바르게 공유되어 있는지 확인하세요.
- **컨테이너 시작 실패**: 포트 충돌 문제일 수 있습니다. `docker ps -a`로 실행 중인 컨테이너를 확인하세요.
- **Nginx 설정 오류**: Nginx 로그를 확인하거나 `docker logs milvus-nginx`를 실행하세요.

#### 문제 해결 단계
1. 로그 확인: `docker logs [컨테이너명]`
2. Nginx 설정 확인: `docker exec -it milvus-nginx cat /etc/nginx/conf.d/server_base.conf`
3. 소켓 확인: `docker exec -it milvus-nginx ls -la /tmp/`
4. 시스템 재시작: `./scripts/cleanup.sh`를 실행한 후 `./scripts/setup.sh full`로 다시 시작

### 11. 오프라인 모드 설정

RAG 시스템을 오프라인 환경에서 실행하려면 다음과 같이 설정을 변경해야 합니다:

#### 11.1 오프라인 모드 활성화

`rag/Dockerfile`에서 다음 환경변수를 수정합니다:

```dockerfile
# 오프라인 모드 비활성화 (기본값)
ENV TRANSFORMERS_OFFLINE=0

# 오프라인 모드 활성화
# ENV TRANSFORMERS_OFFLINE=1
```

오프라인 모드 활성화(`TRANSFORMERS_OFFLINE=1`)로 설정하면, Hugging Face 모델을 온라인에서 다운로드하지 않고 로컬에 이미 다운로드된 모델만 사용합니다. 
배포 환경에서는 이 값을 `1`로 설정하는 것이 권장됩니다.

#### 11.2 모델 사전 다운로드

오프라인 모드에서는 모든 필요한 모델 파일을 사전에 다운로드해야 합니다. 개발 과정에서는 `TRANSFORMERS_OFFLINE=0`으로 설정하고 필요한 모델을 한 번 로드하여 캐시에 저장해 둔 후, 배포 시 `TRANSFORMERS_OFFLINE=1`로 변경하는 것이 좋습니다.