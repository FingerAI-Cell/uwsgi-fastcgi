# uwsgi-fastcgi
A FastCGI-based setup for running Flask with uWSGI and Nginx using Docker Compose.

> 현재 staging 브랜치에서 개발 중입니다.

컨테이너가 실행된 후에 수동으로 pip install numpy==1.24.4를 실행

## 사전 준비사항
#### 1. Docker 및 Docker Compose 설치
- Windows: Docker Desktop 설치
- macOS: Docker Desktop 설치
- Linux: Docker Engine 및 Docker Compose 설치

## 사용 방법

### 1. 자동화 스크립트 사용 (권장)

시스템 셋업과 실행을 자동화하는 스크립트를 제공합니다. 스크립트는 자동으로 필요한 네트워크를 생성합니다.

#### Linux/macOS:
```bash
# 통합 서비스 시작
$ ./scripts/setup.sh

# RAG 서비스만 시작
$ ./scripts/setup-rag.sh

# Reranker 서비스만 시작
$ ./scripts/setup-reranker.sh

# 통합 서비스 종료
$ ./scripts/shutdown.sh

# RAG 서비스만 종료
$ ./scripts/shutdown-rag.sh

# Reranker 서비스만 종료
$ ./scripts/shutdown-reranker.sh

# 도커 리소스 완전 정리 (모든 컨테이너, 관련 이미지, 볼륨, 네트워크 삭제)
$ ./scripts/cleanup.sh
```

#### Windows:
```
# 통합 서비스 시작
$ .\scripts\windows\setup.bat

# RAG 서비스만 시작
$ .\scripts\windows\setup-rag.bat

# Reranker 서비스만 시작
$ .\scripts\windows\setup-reranker.bat

# 통합 서비스 종료
$ .\scripts\windows\shutdown.bat

# RAG 서비스만 종료
$ .\scripts\windows\shutdown-rag.bat

# Reranker 서비스만 종료
$ .\scripts\windows\shutdown-reranker.bat

# 도커 리소스 완전 정리 (모든 컨테이너, 관련 이미지, 볼륨, 네트워크 삭제)
$ .\scripts\windows\cleanup.bat
```

#### 스크립트 디렉토리 구조
스크립트 파일은 다음과 같이 구성되어 있습니다:
```
(루트 디렉토리)
├── scripts/                  - 스크립트 디렉토리
│   ├── setup.sh              - 통합 서비스 시작 스크립트
│   ├── setup-rag.sh          - RAG 서비스 시작 스크립트
│   ├── setup-reranker.sh     - Reranker 서비스 시작 스크립트
│   ├── shutdown.sh           - 통합 서비스 종료 스크립트
│   ├── shutdown-rag.sh       - RAG 서비스 종료 스크립트
│   ├── shutdown-reranker.sh  - Reranker 서비스 종료 스크립트
│   ├── cleanup.sh            - 도커 리소스 정리 스크립트
│   └── windows/              - Windows용 배치 파일
│       ├── setup.bat         - 통합 서비스 시작 스크립트 (Windows)
│       ├── setup-rag.bat     - RAG 서비스 시작 스크립트 (Windows)
│       ├── setup-reranker.bat - Reranker 서비스 시작 스크립트 (Windows)
│       ├── shutdown.bat      - 통합 서비스 종료 스크립트 (Windows)
│       ├── shutdown-rag.bat  - RAG 서비스 종료 스크립트 (Windows)
│       ├── shutdown-reranker.bat - Reranker 서비스 종료 스크립트 (Windows)
│       └── cleanup.bat       - 도커 리소스 정리 스크립트 (Windows)
├── docker-compose.yml        - 통합 서비스 구성
├── docker-compose-rag.yml    - RAG 서비스 구성
└── docker-compose-reranker.yml - Reranker 서비스 구성
```

### 2. 수동으로 서비스 실행 (고급 사용자)

#### 2.0 Docker 네트워크 생성
프로젝트를 수동으로 실행하기 전에 먼저 Docker 네트워크를 생성해야 합니다:
```bash
$ docker network create rag_network
```

#### 2.1 통합 서비스 실행
모든 서비스(RAG, Reranker, API Gateway)를 한 번에 실행합니다:
```bash
# 통합 서비스 빌드 및 시작
$ docker compose up -d --build

# 로그 확인
$ docker compose logs -f
```

#### 2.2 개별 서비스 실행
각 서비스를 따로 실행할 수도 있습니다:

##### RAG 서비스만 실행
```bash
# RAG 서비스 시작
$ docker compose -f docker-compose-rag.yml up -d --build

# 로그 확인
$ docker compose -f docker-compose-rag.yml logs -f
```

##### Reranker 서비스만 실행
```bash
# Reranker 서비스 시작
$ docker compose -f docker-compose-reranker.yml up -d --build

# 로그 확인
$ docker compose -f docker-compose-reranker.yml logs -f
```

#### 2.3 서비스 중지
```bash
# 통합 서비스 중지
$ docker compose down

# 개별 서비스 중지
$ docker compose -f docker-compose-rag.yml down
$ docker compose -f docker-compose-reranker.yml down
```

### 3. 서비스 상태 확인
```bash
# 실행 중인 컨테이너 확인
$ docker ps

# 특정 서비스의 로그 확인
$ docker logs milvus-rag
$ docker logs milvus-reranker
$ docker logs api-gateway
$ docker logs unified-nginx  # 통합 Nginx 컨테이너
```

### 4. 서비스 접근

#### 4.1 통합 서비스 접근 (docker-compose.yml 사용 시)
- 통합 Nginx: **http://localhost:80**
  - RAG 서비스: `/` 경로 (예: http://localhost/search)
  - Reranker 서비스: `/reranker/` 경로 (예: http://localhost/reranker/rerank)
  - API Gateway: `/api/` 경로 (예: http://localhost/api/enhanced-search)
- 서비스 직접 접근(내부 개발용):
  - API Gateway: http://localhost:3000
  - RAG 서비스: http://localhost:5000
  - Reranker 서비스: http://localhost:8000

#### 4.2 개별 서비스 접근 (개별 compose 파일 사용 시)
- RAG 서비스: **http://localhost:80** (docker-compose-rag.yml)
- Reranker 서비스: **http://localhost:8080/reranker/** (docker-compose-reranker.yml)

> **중요**: 통합 서비스와 개별 서비스는 동시에 실행하지 마세요. 포트 충돌이 발생합니다.
> 특히 통합 서비스와 RAG 서비스는 모두 80번 포트를 사용하기 때문에 동시에 실행할 수 없습니다.

#### 4.3 공통 서비스
- Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)

### 5. 데이터 관리
모델 파일 및 데이터는 다음 경로에 저장됩니다:

#### 데이터 저장 위치
- 워크스페이스 데이터: `./workspace`
- 로그 데이터: `./logs`
- 기타 데이터: `./data`
- Reranker 모델: `./reranker/models`

#### 볼륨 관리
데이터를 초기화하거나 볼륨을 삭제하려면:
```bash
# 컨테이너와 볼륨 모두 삭제
$ docker compose down -v

# 특정 볼륨 삭제
$ docker volume rm uwsgi-fastcgi_shared_tmp
```

#### 첫 실행 시 주의사항
- 첫 실행 시 Reranker 모델을 다운로드하므로 시간이 소요될 수 있습니다.
- 모델 다운로드가 완료될 때까지 서비스가 완전히, 정상적으로 동작하지 않을 수 있습니다.
- 다운로드 진행 상황은 `docker logs milvus-reranker`로 확인할 수 있습니다.

### 6. Nginx 설정

#### 6.1 통합 서비스 Nginx 설정 (docker-compose.yml)
- 통합 Nginx 설정: `./default.conf`

#### 6.2 개별 서비스 Nginx 설정
- RAG 서비스: `./rag/default.conf`
- Reranker 서비스: `./reranker/default.conf`

서비스별로 설정을 수정해야 할 경우:
```bash
# 통합 Nginx 컨테이너 접속
$ docker exec -it unified-nginx /bin/bash

# RAG Nginx 컨테이너 접속 (개별 실행 시)
$ docker exec -it milvus-nginx /bin/bash

# Reranker Nginx 컨테이너 접속 (개별 실행 시)
$ docker exec -it reranker-nginx /bin/bash

# 설정 파일 편집
$ apt-get update -y
$ apt install vim -y
$ cd /etc/nginx/conf.d
$ vim default.conf
```

Nginx 재시작은 스크립트를 사용하는 것이 좋지만, 수동으로 할 경우:
```bash
# 현재 디렉토리에서 실행
# Nginx 재시작 (통합 서비스)
$ docker compose restart nginx

# Nginx 재시작 (RAG 서비스)
$ docker compose -f docker-compose-rag.yml restart nginx

# Nginx 재시작 (Reranker 서비스)
$ docker compose -f docker-compose-reranker.yml restart nginx
```

### 7. 서비스 동작 확인
```bash
# 통합 서비스 상태 확인
$ curl http://localhost

# API Gateway 상태 확인 (통합 서비스)
$ curl http://localhost/api/health

# Reranker 상태 확인 (개별 실행 시)
$ curl http://localhost:8080/reranker/health
```

! nginx와 uwsgi는 유닉스 소켓을 통해 통신하기 위해 볼륨을 공유해야 합니다.
Flask는 uwsgi를 통해 실행되므로 uwsgi.ini나 uwsgi 실행 명령이 올바르게 구성되어 있는지 확인하세요.
Nginx에서 502 Bad Gateway 오류가 발생하면 uwsgi.sock이 올바르게 공유되었는지 확인하세요.

### 8. API 호출 예시

#### 8.1 통합 서비스 API 호출 (docker-compose.yml, 포트 80)

##### RAG 서비스 API (루트 경로 /)
```bash
# 기본 검색 API (GET)
curl -X GET "http://localhost/search?query_text=인공지능&top_k=5&domain=news"

# 필터링을 포함한 검색 API
curl -X GET "http://localhost/search?query_text=메타버스&top_k=5&domain=tech&author=삼성전자&start_date=20230101&end_date=20231231"

# 문서 삽입 API (POST)
curl -X POST http://localhost/insert \
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
curl -X DELETE "http://localhost/delete?date=20220804&title=메타버스%20뉴스&author=삼성전자&domain=news"

# 문서 조회 API (GET)
curl -X GET "http://localhost/document?doc_id=20220804-메타버스%20뉴스-삼성전자"

# 컬렉션 정보 조회 API (GET)
curl -X GET "http://localhost/data/show?collection_name=news"
```

##### Reranker 서비스 API (/reranker/ 경로)
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
```

##### API Gateway 서비스 (/api/ 경로)
```bash
# 통합 검색 API (GET)
curl -X GET "http://localhost/api/enhanced-search?query_text=인공지능&top_k=5&raw_results=20&domain=news"

# 필터링을 포함한 통합 검색 API
curl -X GET "http://localhost/api/enhanced-search?query_text=메타버스&top_k=5&raw_results=20&domain=tech&author=삼성전자&start_date=20230101&end_date=20231231"

# API Gateway 상태 확인
curl -X GET "http://localhost/api/health"
```

#### 8.2 개별 서비스 API 호출 (개별 compose 파일)

##### RAG 서비스 API (포트 80, docker-compose-rag.yml)
```bash
# 기본 검색 API (GET)
curl -X GET "http://localhost/search?query_text=인공지능&top_k=5&domain=news"
```

##### Reranker 서비스 API (포트 8080, docker-compose-reranker.yml)
```bash
# Reranker API
curl -X POST http://localhost:8080/reranker/rerank?top_k=5 \
  -H "Content-Type: application/json" \
  -d '{
    "query": "인공지능 최신 기술",
    "results": [
      {"text": "인공지능 기술의 최신 발전 동향에 관한 문서입니다."},
      {"text": "머신러닝 알고리즘의 최근 연구 결과에 대한 분석입니다."},
      {"text": "인공지능이 다양한 산업에 미치는 영향에 대한 연구입니다."}
    ]
  }'

# 상태 확인 API
curl -X GET http://localhost:8080/reranker/health
```

### 9. 시스템 아키텍처

#### 서비스 구성
- **Milvus 백엔드**: 벡터 검색을 위한 데이터베이스 (etcd, minio, standalone)
- **RAG 서비스**: 검색 요청을 처리하고 Milvus에서 데이터를 검색
- **Reranker 서비스**: 검색 결과를 재랭킹하여 관련성이 높은 순서로 정렬
- **API Gateway**: RAG와 Reranker 서비스를 연결하여 통합 검색 결과 제공
- **Nginx**: 모든 서비스에 대한 접근을 관리하는 단일 웹 서버

#### 통합 검색 파이프라인
1. 사용자가 `/api/enhanced-search` 엔드포인트로 검색 쿼리 전송
2. API Gateway가 RAG 서비스에 검색 요청을 전달하여 초기 검색 결과 획득 (기본 20개)
3. API Gateway가 Reranker 서비스에 결과를 전달하여 관련성에 따라 재정렬
4. 상위 결과(기본 5개)만 사용자에게 반환

### 10. 크로스 플랫폼 호환성 노트
#### 1. 파일 경로
- Windows에서는 파일 경로 구분자로 `\`를 사용하지만, Docker 내부에서는 항상 리눅스 방식인 `/`를 사용합니다.
- 프로젝트에서 상대 경로를 사용하므로 대부분의 OS에서 문제없이 작동해야 합니다.

#### 2. 파일 권한
- Windows에서 작업 후 Linux/macOS에 배포할 경우, 실행 권한이 없을 수 있습니다.
- 필요한 경우 Linux/macOS에서 다음 명령을 실행합니다:
```bash
$ chmod +x setup.sh shutdown.sh setup-rag.sh shutdown-rag.sh setup-reranker.sh shutdown-reranker.sh cleanup.sh
$ chmod +x scripts/*.sh
```

#### 3. 라인 엔딩
- Windows와 Linux/macOS는 라인 엔딩이 다릅니다 (CRLF vs LF).
- Git 설정에서 `core.autocrlf`를 true나 input으로 설정하여 문제를 방지할 수 있습니다:
```bash
$ git config --global core.autocrlf input  # Linux/macOS
$ git config --global core.autocrlf true   # Windows
```

#### 4. 문제 해결
- 소켓 연결 오류가 발생하면 uwsgi.sock 파일이 올바르게 공유되었는지 확인하세요.
- 권한 문제가 발생하면 Docker 볼륨과 파일 권한을 확인하세요.
- 스크립트 실행 오류가 발생하면 실행 권한이 있는지 확인하세요.

## 6. 오프라인 모드 설정

RAG 시스템을 오프라인 환경에서 실행하려면 다음과 같이 설정을 변경해야 합니다:

### 6.1 오프라인 모드 활성화

`rag/Dockerfile`에서 다음 환경변수를 수정합니다:

```dockerfile
# 오프라인 모드 비활성화 (기본값)
ENV TRANSFORMERS_OFFLINE=0

# 오프라인 모드 활성화
# ENV TRANSFORMERS_OFFLINE=1
```

오프라인 모드 활성화(`TRANSFORMERS_OFFLINE=1`)로 설정하면, Hugging Face 모델을 온라인에서 다운로드하지 않고 로컬에 이미 다운로드된 모델만 사용합니다. 
배포 환경에서는 이 값을 `1`로 설정하는 것이 권장됩니다.

### 6.2 모델 사전 다운로드

오프라인 모드에서는 모든 필요한 모델 파일을 사전에 다운로드해야 합니다. 개발 과정에서는 `TRANSFORMERS_OFFLINE=0`으로 설정하고 필요한 모델을 한 번 로드하여 캐시에 저장해 둔 후, 배포 시 `TRANSFORMERS_OFFLINE=1`로 변경하는 것이 좋습니다.
