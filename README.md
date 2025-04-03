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

### Docker compose
#### 1. Customize docker-compose.yml if needed 
- Ensure that the Flask and Nginx services share the volume (shared_tmp) and network (flask_network).
- Modify docker-compose.yml as necessary.


#### 2. Start Containers 
```bash
$ docker-compose up -d --build
```
- The -d flag runs the containers in the background.
- The --build flag rebuilds the containers if changes are made


#### 3. Editing Nginx Configuration
we need to edit default.conf file of nginx in container 
```bash
$ docker exec -it flask-nginx /bin/bash 
```
Edit default.conf
```bash
$ apt-get update -y
$ apt install vim -y
$ cd /etc/nginx/conf.d
$ vi default.conf
```
Replace the content of default.conf with the configuration from this repository.
Then restart nginx 
```bash
$ docker-compose restart nginx
```


### Verify the Setup 
```
$ curl http://localhost
```
Expected output: Hello, FastCGI is working!

! nginx and uwsgi must share a volume to connect via a Unix socket.
Flask is executed through uwsgi, so ensure that uwsgi.ini or the uwsgi execution command is correctly configured.
If Nginx returns a 502 Bad Gateway error, verify that uwsgi.sock is properly shared.

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
