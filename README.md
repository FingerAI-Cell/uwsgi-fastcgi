# uwsgi-fastcgi
A FastCGI-based setup for running Flask with uWSGI and Nginx using Docker Compose.

## 사전 준비사항
#### 1. Docker 및 Docker Compose 설치
- Windows: Docker Desktop 설치
- macOS: Docker Desktop 설치
- Linux: Docker Engine 및 Docker Compose 설치

#### 2. 네트워크 생성
프로젝트를 실행하기 전에 필요한 Docker 네트워크를 생성합니다:
```bash
$ docker network create rag_network
$ docker network create milvus-net
```

## 사용 방법

### 1. RAG 서비스 실행
RAG 서비스는 Milvus와 연동되는 메인 서비스입니다.

#### 서비스 시작
```bash
# 서비스 빌드 및 시작
$ docker-compose up -d --build

# 로그 확인
$ docker-compose logs -f
```

#### 서비스 중지
```bash
# 서비스 중지
$ docker-compose down
```

### 2. Reranker 서비스 실행
Reranker 서비스는 검색 결과의 순위를 재조정하는 서비스입니다.

#### 서비스 시작
```bash
# 서비스 빌드 및 시작
$ docker-compose -f docker-compose-reranker.yml up -d --build

# 로그 확인
$ docker-compose -f docker-compose-reranker.yml logs -f
```

#### 서비스 중지
```bash
# 서비스 중지
$ docker-compose -f docker-compose-reranker.yml down
```

### 3. 전체 서비스 한 번에 실행
두 서비스를 함께 실행하려면:

```bash
# 두 서비스 모두 시작
$ docker-compose up -d --build && docker-compose -f docker-compose-reranker.yml up -d --build

# 두 서비스 모두 중지
$ docker-compose down && docker-compose -f docker-compose-reranker.yml down
```

### 4. 서비스 상태 확인
```bash
# 실행 중인 컨테이너 확인
$ docker ps

# 특정 서비스의 로그 확인
$ docker logs milvus-rag
$ docker logs milvus-reranker
```

### 5. 서비스 접근
- RAG 서비스: http://localhost:80
- Reranker 서비스: http://localhost:8080
- Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)

### 6. 데이터 관리
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
$ docker-compose down -v

# 특정 볼륨 삭제
$ docker volume rm uwsgi-fastcgi_shared_tmp
```

#### 첫 실행 시 주의사항
- 첫 실행 시 Reranker 모델을 다운로드하므로 시간이 소요될 수 있습니다.
- 모델 다운로드가 완료될 때까지 서비스가 완전히, 정상적으로 동작하지 않을 수 있습니다.
- 다운로드 진행 상황은 `docker logs milvus-reranker`로 확인할 수 있습니다.

### 7. Nginx 설정
각 서비스별로 별도의 Nginx 설정 파일이 있습니다:
- RAG 서비스: `./rag/default.conf`
- Reranker 서비스: `./reranker/default.conf`
- 루트 디렉토리의 `default.conf`는 두 설정을 합친 참조용 파일입니다.

서비스별로 설정을 수정해야 할 경우:
```bash
# RAG Nginx 컨테이너 접속
$ docker exec -it milvus-nginx /bin/bash

# Reranker Nginx 컨테이너 접속
$ docker exec -it reranker-nginx /bin/bash

# 설정 파일 편집
$ apt-get update -y
$ apt install vim -y
$ cd /etc/nginx/conf.d
$ vim default.conf

# Nginx 재시작
# RAG 서비스용
$ docker-compose restart nginx

# Reranker 서비스용
$ docker-compose -f docker-compose-reranker.yml restart nginx
```

### 8. 서비스 동작 확인
```bash
# RAG 서비스 확인
$ curl http://localhost

# Reranker 서비스 확인 
$ curl http://localhost:8080/health
```

! nginx와 uwsgi는 유닉스 소켓을 통해 통신하기 위해 볼륨을 공유해야 합니다.
Flask는 uwsgi를 통해 실행되므로 uwsgi.ini나 uwsgi 실행 명령이 올바르게 구성되어 있는지 확인하세요.
Nginx에서 502 Bad Gateway 오류가 발생하면 uwsgi.sock이 올바르게 공유되었는지 확인하세요.

### 9. API 호출 예시

#### RAG 서비스 API (포트 80)
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

# 서비스 상태 확인 API (GET)
curl -X GET http://localhost/
```

#### Reranker 서비스 API (포트 8080)
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

# Batch Reranking API
curl -X POST http://localhost:8080/reranker/batch_rerank?top_k=3 \
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
curl -X GET http://localhost:8080/reranker/health
```

### 크로스 플랫폼 호환성 노트
#### 1. 파일 경로
- Windows에서는 파일 경로 구분자로 `\`를 사용하지만, Docker 내부에서는 항상 리눅스 방식인 `/`를 사용합니다.
- 프로젝트에서 상대 경로를 사용하므로 대부분의 OS에서 문제없이 작동해야 합니다.

#### 2. 파일 권한
- Windows에서 작업 후 Linux/macOS에 배포할 경우, 실행 권한이 없을 수 있습니다.
- 필요한 경우 Linux/macOS에서 다음 명령을 실행합니다:
```bash
$ chmod +x *.sh  # 셸 스크립트에 실행 권한 부여
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
