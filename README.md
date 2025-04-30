# uwsgi-fastcgi
FastCGI 기반의 uWSGI와 Nginx를 사용하여 Docker Compose로 구성된 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 사전 준비사항
#### 1. Docker 및 Docker Compose 설치
- Windows: Docker Desktop 설치
- macOS: Docker Desktop 설치
- Linux: Docker Engine 및 Docker Compose 설치

#### 2. Milvus 데이터 경로 설정
서비스를 시작하기 전에 Milvus 데이터 저장 경로를 설정할 수 있습니다. `setup.sh` 실행 시 다음과 같은 옵션이 있습니다:

1. 기본 경로 사용 (`/var/lib/milvus-data`)
   - 그냥 엔터를 누르면 기본 경로가 사용됩니다.

2. 사용자 지정 경로 설정
   - 절대 경로 (예: `/data/milvus`)
   - 현재 위치 기준 상대 경로 (예: `./data/milvus`)
   - 프로젝트 루트 기준 상대 경로 (예: `../data/milvus`)

설정된 경로는 `config/storage.json`에 저장되며, 이후 `purge_volumes.sh` 실행 시에도 동일한 경로가 사용됩니다.

**주의사항**:
- 상대 경로 입력 시 현재 스크립트 실행 위치를 기준으로 합니다.
- 데이터 관리를 위해 절대 경로 사용을 권장합니다.
- 설정 파일(`config/storage.json`)은 git에 포함되지 않습니다.
- 처음 실행 시 기본 설정 파일이 자동으로 생성됩니다.

#### 3. 볼륨 디렉토리 설정
시스템은 데이터 저장을 위해 두 가지 볼륨 위치를 사용합니다:

1. **내부 볼륨 `/var/lib/milvus-data/`** ⚠️ **(중요: Milvus DB 데이터 저장 위치)**
   - Milvus 관련 컨테이너(etcd, minio, standalone)의 데이터 저장
   - 권한 문제를 방지하기 위해 공유 폴더가 아닌 호스트 내부에 저장
   - etcd 서비스는 엄격한 권한 요구사항(700)이 있음

2. **로컬 볼륨 `./volumes/`** (로그 및 설정 파일용)
   - 설정 파일 및 로그 파일 저장
   - 개발 편의성을 위해 공유 폴더에 유지

**VirtualBox 공유 폴더 주의사항**:
공유 폴더는 Linux 권한 시스템을 완전히 지원하지 않을 수 있습니다. 따라서 권한 설정이 중요한 데이터베이스 컨테이너는 호스트 내부 볼륨을 사용합니다.

**Linux 환경 설정**:
```bash
# 내부 볼륨 디렉토리 생성 (스크립트가 자동 실행)
sudo mkdir -p /var/lib/milvus-data/{etcd,minio,milvus,logs/{etcd,minio,milvus}}
sudo chown -R $(whoami):$(whoami) /var/lib/milvus-data
chmod -R 700 /var/lib/milvus-data/etcd
```

**Windows WSL 환경 설정**:
Windows 환경에서는 WSL 터미널에서 다음 명령을 실행해야 합니다:
```bash
# WSL 터미널에서 실행
wsl -d Ubuntu sudo mkdir -p /var/lib/milvus-data/{etcd,minio,milvus,logs/{etcd,minio,milvus}}
wsl -d Ubuntu sudo chown -R $(whoami):$(whoami) /var/lib/milvus-data
wsl -d Ubuntu chmod -R 700 /var/lib/milvus-data/etcd
```

**권한 설정 검증**:
setup.sh 스크립트는 볼륨 디렉토리 생성 후 자동으로 권한을 검증합니다:
```
내부 볼륨 권한 설정 확인 중...
etcd 디렉토리 권한:
drwx------ 5 사용자 사용자그룹 4096 월 일 시간 /var/lib/milvus-data/etcd
소유권 확인:
drwxr-xr-x 7 사용자 사용자그룹 4096 월 일 시간 /var/lib/milvus-data
```
etcd 디렉토리의 권한이 700(drwx------)으로 설정되었는지 확인하세요. 이 설정이 없으면 etcd 서비스가 정상 작동하지 않을 수 있습니다.

만약 권한 설정에 문제가 발생한다면, setup.sh 스크립트를 다시 실행하거나 필요한 경우에만 주의해서 purge_volumes.sh 스크립트를 사용하세요. 주의: purge_volumes.sh는 데이터를 삭제할 수 있으므로 중요한 데이터는 먼저 백업하세요.

## 시스템 요구사항

전체 시스템(all 프로필)을 실행하기 위한 최소 요구사항입니다:

| 구분 | 최소 요구사항 | 권장 사양 | 비고 |
|------|-------------|-----------|------|
| **CPU** | 2코어 | 4코어 이상 | Reranker 모델 실행 시 더 많은 코어가 성능 향상에 도움됨 |
| **메모리(RAM)** | 8GB | 16GB | 총합: 최소 8GB 필요 |
| └ Milvus | 4GB | 8GB | 벡터 검색 엔진 |
| └ Reranker | 2GB | 4GB | 모델 로딩 시 더 많은 메모리 필요 가능 |
| └ 기타 서비스 | 2GB | 4GB | Etcd, MinIO, Nginx, RAG 서비스 등 |
| **디스크 공간** | 30GB | 50GB 이상 | 총합: 최소 30GB 필요 (실측 기준) |
| └ RAG 이미지 | 19GB | 20GB | PyTorch 및 관련 라이브러리 포함 |
| └ Reranker 이미지 | 2GB | 3GB | 모델 추론 관련 라이브러리 포함 |
| └ 기타 이미지 | 2GB | 3GB | Nginx, Milvus, Etcd, MinIO 등 |
| └ 컨테이너 런타임 | 5GB | 8GB | 컨테이너 실행 시 추가 공간 |
| └ 벡터 데이터 | 2GB | 10GB+ | 데이터 양에 따라 크게 증가할 수 있음 |
| └ 빌드 캐시 | 1GB | 3GB | 빌드 및 실행 캐시 |
| └ 로그 및 기타 | 3GB | 8GB | 로그, 임시 파일 등 |

**참고사항**:
- 데이터 양이 증가함에 따라 디스크 공간 요구사항도 증가합니다.
- 개발 환경에서는 최소 요구사항보다 낮은 사양으로도 작동할 수 있으나, 프로덕션 환경에서는 권장 사양 이상을 권장합니다.
- 가상 머신이나 컨테이너 환경에서 실행 시 적절한 리소스 할당이 필요합니다.

## 프로젝트 구조
```
(루트 디렉토리)
├── nginx/                    - Nginx 설정 파일
│   ├── conf.d/               - Nginx 서버 설정
│   │   └── server_base.conf  - 기본 서버 설정
│   ├── locations-enabled/    - 활성화된 location 설정 (빈 디렉토리)
│   ├── templates/            - location 템플릿 파일
│   │   ├── rag.conf.template - RAG 서비스 템플릿
│   │   ├── prompt.conf.template - Prompt 서비스 템플릿
│   │   └── reranker.conf.template - Reranker 서비스 템플릿
│   └── nginx.conf           - Nginx 기본 설정
├── scripts/                  - 스크립트 디렉토리
│   ├── setup.sh              - 서비스 시작 스크립트
│   ├── cleanup.sh            - 도커 리소스 정리 스크립트
│   ├── purge_volumes.sh      - 로컬 볼륨 완전 제거 스크립트
│   └── shutdown.sh           - 서비스 종료 스크립트
├── volumes/                  - 로컬 로그 저장 디렉토리
│   └── logs/                 - 로그 디렉토리
│       ├── nginx/            - Nginx 로그
│       ├── rag/              - RAG 서비스 로그
│       ├── reranker/         - Reranker 서비스 로그
│       ├── prompt/           - Prompt 서비스 로그
│       └── ollama/           - Ollama 서비스 로그
│   └── ollama/               - Ollama 모델 저장소
├── /var/lib/milvus-data/    - 호스트 내부 데이터 볼륨
│   ├── etcd/                 - Etcd 데이터 
│   ├── minio/                - MinIO 데이터
│   ├── milvus/               - Milvus 데이터
│   └── logs/                 - DB 관련 로그
│       ├── etcd/             - Etcd 로그
│       ├── minio/            - MinIO 로그
│       └── milvus/           - Milvus 로그
├── rag/                      - RAG 서비스 디렉토리
├── reranker/                 - Reranker 서비스 디렉토리
├── prompt/                   - Prompt 서비스 디렉토리
├── ollama/                   - Ollama 서비스 디렉토리
│   ├── Dockerfile            - Ollama 도커 이미지 설정
│   ├── init.sh               - Ollama 초기화 스크립트
│   └── models.txt            - Ollama 모델 목록
├── milvus/                   - Milvus 설정 디렉토리
├── flashrank/                - Flashrank 라이브러리
├── docker-compose.yml        - Docker Compose 설정
└── .env                      - 환경 변수 설정
```

## 사용 방법

다음 명령어로 각 서비스를 시작할 수 있습니다:

```bash
# 모든 서비스 시작 (CPU 모드)
./scripts/setup.sh all

# 모든 서비스 시작 (GPU 모드)
./scripts/setup.sh all-gpu

# RAG 및 Reranker 서비스만 시작
./scripts/setup.sh rag_reranker

# Prompt 서비스만 시작
./scripts/setup.sh prompt

# 앱 서비스만 시작 (CPU 모드, DB 제외)
./scripts/setup.sh app-only

# 앱 서비스만 시작 (GPU 모드, DB 제외)
./scripts/setup.sh app-only-gpu

# Ollama 서비스만 시작 (CPU 모드)
./scripts/setup.sh ollama

# Ollama 서비스만 시작 (GPU 모드)
./scripts/setup.sh ollama-gpu

# Prompt 및 Ollama 서비스 조합 시작 (CPU 모드)
./scripts/setup.sh prompt_ollama

# Prompt 및 Ollama 서비스 조합 시작 (GPU 모드)
./scripts/setup.sh prompt_ollama-gpu
```

#### CPU 및 GPU 모드 차이점
- **CPU 모드**: gemma:2b와 같은 가벼운 모델만 사용 가능하며, 응답이 느립니다.
- **GPU 모드**: 더 큰 모델(mistral, llama3 등)을 사용할 수 있으며, 응답 속도가 빠릅니다.
- 개발이나 테스트 환경에서는 CPU 모드를, 프로덕션 환경에서는 GPU 모드를 권장합니다.
- RAM이 16GB 이상 있으면 CPU에서도 mistral 모델(7B)을 실행할 수 있지만, 매우 느립니다.

#### GPU 요구사항
GPU 모드를 사용하려면:
1. NVIDIA GPU가 설치된 시스템이 필요합니다.
2. NVIDIA 드라이버가 설치되어 있어야 합니다.
3. Docker에 NVIDIA 컨테이너 툴킷이 설치되어 있어야 합니다.
4. NVIDIA 컨테이너 런타임이 Docker의 기본 런타임으로 설정되어 있어야 합니다.

### 1. 자동화 스크립트 사용

시스템 셋업과 실행을 자동화하는 스크립트를 제공합니다.

#### Linux/macOS:
```bash
# 모든 서비스 시작 (RAG + Reranker + Prompt + Ollama(CPU) + DB)
$ ./scripts/setup.sh all

# 모든 서비스 시작 (RAG + Reranker + Prompt + Ollama(GPU) + DB)
$ ./scripts/setup.sh all-gpu

# RAG 서비스만 시작 (DB 포함)
$ ./scripts/setup.sh rag

# Prompt 서비스만 시작
$ ./scripts/setup.sh prompt

# RAG + Reranker 시작 (DB 포함)
$ ./scripts/setup.sh rag-reranker

# 데이터베이스 서비스만 시작 (Milvus, Etcd, MinIO)
$ ./scripts/setup.sh db

# 애플리케이션 서비스만 시작 (RAG, Reranker, Prompt, Ollama(CPU) - DB 제외)
# DB가 이미 실행 중일 때 코드 변경 후 사용
$ ./scripts/setup.sh app-only

# 애플리케이션 서비스만 시작 (RAG, Reranker, Prompt, Ollama(GPU) - DB 제외)
# DB가 이미 실행 중일 때 코드 변경 후 사용
$ ./scripts/setup.sh app-only-gpu

# 서비스 종료
$ ./scripts/shutdown.sh [all|rag|reranker|prompt|rag-reranker|db|app-only]

# 도커 리소스 완전 정리 (모든 컨테이너, 관련 이미지, 볼륨, 네트워크 삭제)
$ ./scripts/cleanup.sh

# 로컬 볼륨 디렉토리 완전 제거 (모든 데이터 삭제)
$ ./scripts/purge_volumes.sh
```

#### Windows:
```powershell
# 모든 서비스 시작 (RAG + Reranker + Prompt + Ollama(CPU) + DB)
.\scripts\windows\setup.bat all

# 모든 서비스 시작 (RAG + Reranker + Prompt + Ollama(GPU) + DB)
.\scripts\windows\setup.bat all-gpu

# RAG 서비스만 시작 (DB 포함)
.\scripts\windows\setup.bat rag

# Reranker 서비스만 시작
.\scripts\windows\setup.bat reranker

# Prompt 서비스만 시작
.\scripts\windows\setup.bat prompt

# RAG + Reranker 시작 (DB 포함)
.\scripts\windows\setup.bat rag-reranker

# 데이터베이스 서비스만 시작 (Milvus, Etcd, MinIO)
.\scripts\windows\setup.bat db

# 애플리케이션 서비스만 시작 (RAG, Reranker, Prompt, Ollama(CPU) - DB 제외)
.\scripts\windows\setup.bat app-only

# 애플리케이션 서비스만 시작 (RAG, Reranker, Prompt, Ollama(GPU) - DB 제외)
.\scripts\windows\setup.bat app-only-gpu

# Ollama 서비스만 시작 (CPU 모드)
.\scripts\windows\setup.bat ollama

# Ollama 서비스만 시작 (GPU 모드)
.\scripts\windows\setup.bat ollama-gpu

# Prompt 및 Ollama 서비스 조합 시작 (CPU 모드)
.\scripts\windows\setup.bat prompt_ollama

# Prompt 및 Ollama 서비스 조합 시작 (GPU 모드)
.\scripts\windows\setup.bat prompt_ollama-gpu

# 서비스 종료
.\scripts\windows\shutdown.bat [all|rag|reranker|prompt|rag-reranker|db|app-only]

# 도커 리소스 완전 정리
.\scripts\windows\cleanup.bat

# 로컬 볼륨 디렉토리 완전 제거
.\scripts\windows\purge_volumes.bat
```

### 2. 서비스 상태 확인
```bash
# 실행 중인 컨테이너 확인
$ docker ps

# 특정 서비스의 로그 확인
$ docker logs milvus-rag
$ docker logs milvus-reranker
$ docker logs milvus-prompt
$ docker logs milvus-nginx
```

### 3. 서비스 접근

#### 서비스 엔드포인트
- RAG 서비스: **http://localhost/rag/**
- Reranker 서비스: **http://localhost/reranker/**
- Prompt 서비스: **http://localhost/prompt/**
- 통합 요약 API: **http://localhost/prompt/summarize**
- 통합 검색 API: **http://localhost/reranker/enhanced-search**
- Milvus UI: **http://localhost:9001** (사용자: minioadmin, 비밀번호: minioadmin)

### 4. Nginx 설정

#### 설정 파일 구조
- 서버 설정: `nginx/conf.d/server_base.conf`
- Location 템플릿: `nginx/templates/`
  - RAG 서비스: `nginx/templates/rag.conf.template`
  - Reranker 서비스: `nginx/templates/reranker.conf.template`
  - Prompt 서비스: `nginx/templates/prompt.conf.template`
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
$ curl http://localhost/rag/

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
```

#### 통합 검색 API (Enhanced Search)
```bash
# 통합 검색 API (RAG + Reranker)
curl -X GET "http://localhost/reranker/enhanced-search?query_text=인공지능&top_k=5&domain=news"

# 필터링을 포함한 통합 검색 API
curl -X GET "http://localhost/reranker/enhanced-search?query_text=메타버스&top_k=5&domain=tech&author=삼성전자&start_date=20230101&end_date=20231231"

# 상세 파라미터를 포함한 통합 검색 API
curl -X GET "http://localhost/reranker/enhanced-search?query_text=인공지능&top_k=5&search_k=10&rerank_k=5&domain=news"
```

#### Prompt 서비스 API (/prompt/ 경로)
```bash
# 상태 확인 API
curl -X GET "http://localhost/prompt/health"

# 요약 API
curl -X POST http://localhost/prompt/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "query": "인공지능의 최신 기술 동향을 요약해주세요",
    "domain": "tech"
  }'

# 챗봇 API
curl -X POST http://localhost/prompt/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "인공지능이란 무엇인가요?"
  }'
```

### 7. 시스템 아키텍처

#### 서비스 구성
- **Milvus 백엔드**: 벡터 검색을 위한 데이터베이스 (etcd, minio, standalone)
- **RAG 서비스**: 검색 요청을 처리하고 Milvus에서 데이터를 검색
- **Reranker 서비스**: 검색 결과를 재랭킹하여 관련성이 높은 순서로 정렬
- **Prompt 서비스**: 검색 및 재랭킹 결과를 바탕으로 요약 또는 질의응답 생성
- **Nginx**: 모든 서비스에 대한 접근을 관리하는 웹 서버

#### FastCGI 구성
- RAG, Reranker, Prompt 서비스는 모두 uWSGI를 통해 FastCGI 프로토콜로 실행
- Nginx는 Unix 소켓을 통해 uWSGI와 통신
- 각 서비스는 독립적인 Docker 컨테이너에서 실행되지만 소켓을 공유

### 8. 개발 워크플로우

#### 1. 초기 셋업
```bash
# 모든 서비스 시작 (처음 실행 시)
$ ./scripts/setup.sh all
```

#### 2. 특정 서비스만 개발
```bash
# Prompt 서비스만 실행하여 개발할 경우
$ ./scripts/setup.sh prompt

# RAG + Reranker 서비스만 실행할 경우
$ ./scripts/setup.sh rag-reranker
```

#### 3. 코드 변경 후 애플리케이션 재시작
```bash
# DB는 그대로 두고 애플리케이션만 재시작
$ ./scripts/setup.sh app-only
```

#### 4. 데이터베이스만 실행
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

#### 데이터 초기화

시스템 데이터를 완전히 초기화하는 방법은 두 가지가 있습니다:

1. **Docker 리소스만 정리 (볼륨 데이터 유지)**
   ```bash
   # Linux/macOS
   $ ./scripts/cleanup.sh
   
   # Windows
   $ .\scripts\windows\cleanup.bat
   ```
   이 명령은 Docker 컨테이너, 이미지, 볼륨을 삭제하지만, 로컬 파일 시스템의 볼륨 데이터는 그대로 유지됩니다.

2. **로컬 볼륨 디렉토리 완전 제거 (모든 데이터 삭제)**
   ```bash
   # Linux/macOS
   $ ./scripts/purge_volumes.sh
   
   # Windows
   $ .\scripts\windows\purge_volumes.bat
   ```
   이 명령은 로컬 `volumes` 디렉토리를 완전히 삭제하고 빈 디렉토리 구조를 다시 생성합니다. 모든 데이터가 영구적으로 삭제되므로 주의하세요.

> **참고**: Linux/macOS에서 실행 권한 오류가 발생하면 다음 명령으로 실행 권한을 부여하세요:
> ```bash
> $ chmod +x scripts/cleanup.sh scripts/purge_volumes.sh scripts/setup.sh
> ```

**완전한 초기화를 위한 권장 순서**:
1. 먼저 `cleanup.sh`/`cleanup.bat`으로 Docker 리소스 정리
2. 그 다음 `purge_volumes.sh`/`purge_volumes.bat`으로 로컬 볼륨 초기화
3. 마지막으로 `setup.sh all`로 시스템 재시작

### 10. 문제 해결

#### 일반적인 문제
- **502 Bad Gateway**: FastCGI 소켓 연결 문제. 소켓 파일이 올바르게 공유되어 있는지 확인하세요.
- **컨테이너 시작 실패**: 포트 충돌 문제일 수 있습니다. `docker ps -a`로 실행 중인 컨테이너를 확인하세요.
- **Nginx 설정 오류**: Nginx 로그를 확인하거나 `docker logs milvus-nginx`를 실행하세요.

#### 문제 해결 단계
1. 로그 확인: `docker logs [컨테이너명]`
2. Nginx 설정 확인: `docker exec -it milvus-nginx cat /etc/nginx/conf.d/server_base.conf`
3. 소켓 확인: `docker exec -it milvus-nginx ls -la /tmp/`
4. 시스템 재시작: `./scripts/cleanup.sh`를 실행한 후 `./scripts/setup.sh all`로 다시 시작

#### 로그 확인 방법
1. 컨테이너 로그 (실시간):
   ```bash
   # uWSGI 및 애플리케이션 시작 로그
   docker logs -f milvus-prompt
   
   # 상세 애플리케이션 로그
   docker exec milvus-prompt cat /var/log/prompt/app.log
   
   # uWSGI 로그
   docker exec milvus-prompt cat /var/log/prompt/uwsgi.log
   ```

2. 호스트 시스템에서 직접 확인:
   ```bash
   # 애플리케이션 로그
   cat ./volumes/logs/prompt/app.log
   
   # uWSGI 로그
   cat ./volumes/logs/prompt/uwsgi.log
   ```

3. 실시간 로그 모니터링:
   ```bash
   # 애플리케이션 로그 실시간 확인
   tail -f ./volumes/logs/prompt/app.log
   
   # uWSGI 로그 실시간 확인
   tail -f ./volumes/logs/prompt/uwsgi.log
   ```

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

### 12. Ollama 서비스 사용

#### Ollama 서비스 API (직접 호출)
```bash
# 모델 목록 가져오기
curl -X GET "http://localhost:11434/api/tags"

# 모델 다운로드
curl -X POST "http://localhost:11434/api/pull" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "mistral"
  }'

# 텍스트 생성
curl -X POST "http://localhost:11434/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "prompt": "인공지능의 미래에 대해 알려주세요."
  }'
```

#### 프롬프트 서비스와 Ollama 연동 API
```bash
# 모델 목록 가져오기
curl -X GET "http://localhost/prompt/models"

# 챗봇 API
curl -X POST "http://localhost/prompt/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "인공지능이란 무엇인가요?",
    "model": "mistral"
  }'

# 요약 API
curl -X POST "http://localhost/prompt/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "인공지능의 최신 기술 동향을 요약해주세요",
    "domain": "tech"
  }'
```

#### CPU 및 GPU 모드 차이점
- **CPU 모드**: gemma:2b와 같은 가벼운 모델만 사용 가능하며, 응답이 느립니다.
- **GPU 모드**: 더 큰 모델(mistral, llama3 등)을 사용할 수 있으며, 응답 속도가 빠릅니다.
- 개발이나 테스트 환경에서는 CPU 모드를, 프로덕션 환경에서는 GPU 모드를 권장합니다.
- RAM이 16GB 이상 있으면 CPU에서도 mistral 모델(7B)을 실행할 수 있지만, 매우 느립니다.

# Vision 서비스 API (/vision/ 경로)
```bash
# 상태 확인 API
curl -X GET http://localhost/vision/health

# 이미지 분석 API
curl -X POST http://localhost/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/image.jpg",
    "prompt": "이 이미지에 대해 설명해주세요",
    "model": "llama:3.2-11b-vision"
  }'
```

# RAG System Setup

## 모델 파일 설정

### 수동 전송이 필요한 파일
BGE-M3 모델의 다음 파일은 용량 문제로 Git에서 제외되어 있어 수동 전송이 필요합니다:

- `models/bge-m3/pytorch_model.bin` (2.27GB)

### 전송 방법
1. VPN 연결 후 다음 명령어로 전송:
```bash
scp [로컬경로]/models/bge-m3/pytorch_model.bin [서버계정]@[서버경로]/home/[홈디렉토리]/uwsgi-fastcgi/models/bge-m3/
```

2. 전송 후 setup.sh 실행:
```bash
cd /home/jsh0630/uwsgi-fastcgi
./scripts/setup.sh
```