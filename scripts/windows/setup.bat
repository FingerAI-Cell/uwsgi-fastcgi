@echo off
setlocal enabledelayedexpansion

:: 스크립트 시작 메시지 출력
echo === RAG 시스템 셋업 시작 ===

:: 현재 디렉토리 확인 및 루트 디렉토리로 이동
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\.."
cd /d "%ROOT_DIR%"

:: Docker 실행 확인
docker ps >nul 2>&1
if errorlevel 1 (
    echo Error: Docker가 실행 중이지 않습니다.
    echo Docker Desktop을 실행하고 다시 시도하세요.
    exit /b 1
)

:: 필요한 Docker 네트워크 생성 (이미 존재하는 경우 무시)
echo Docker 네트워크 생성 중...
docker network create rag_network 2>nul
if errorlevel 1 (
    echo rag_network가 이미 존재합니다.
)

:: 설정 파일 디렉토리 및 경로 설정
set "CONFIG_DIR=%ROOT_DIR%\config"
set "CONFIG_FILE=%CONFIG_DIR%\storage.json"
set "DEFAULT_MILVUS_PATH=/var/lib/milvus-data"

:: config 디렉토리 생성
if not exist "%CONFIG_DIR%" (
    mkdir "%CONFIG_DIR%"
)

:: 설정 파일이 있으면 읽기
set "MILVUS_PATH=%DEFAULT_MILVUS_PATH%"
if exist "%CONFIG_FILE%" (
    for /f "usebackq tokens=* delims=" %%a in (`type "%CONFIG_FILE%" ^| findstr "milvus_data_path"`) do (
        for /f "tokens=2 delims=:, " %%b in ("%%a") do (
            set "STORED_PATH=%%~b"
            if not "!STORED_PATH!"=="" if not "!STORED_PATH!"=="null" (
                set "MILVUS_PATH=!STORED_PATH!"
            )
        )
    )
)

:: 사용자 입력 안내
echo ============= Milvus 데이터 경로 설정 =============
echo 현재 프로젝트 루트 디렉토리: %CD%
echo 다음과 같은 형식의 경로를 입력할 수 있습니다:
echo 1. 절대 경로 (예: /var/lib/milvus-data)
echo 2. 프로젝트 루트 기준 상대 경로 (예: ./data/milvus)
echo ※ 주의: './data/milvus'와 같이 입력하면 '%CD%\data\milvus'로 처리됩니다.
echo ※ 권장: 데이터 관리를 위해 절대 경로 사용을 권장합니다.
echo ==================================================
set /p "INPUT_PATH=Milvus 데이터 저장 경로를 입력하세요 (기본값: %MILVUS_PATH%): "

:: 입력이 없으면 기본값 사용
if "!INPUT_PATH!"=="" (
    set "INPUT_PATH=%MILVUS_PATH%"
)

:: 상대 경로를 절대 경로로 변환
if "!INPUT_PATH:~0,2!"=="./" (
    :: 디렉토리 부분과 파일명 부분 분리
    for %%i in ("!INPUT_PATH!") do (
        set "DIR_PART=%%~dpi"
        set "BASE_PART=%%~nxi"
    )
    
    :: 상위 디렉토리가 없어도 mkdir로 생성
    mkdir "!DIR_PART!" 2>nul
    
    :: 절대 경로로 변환
    pushd "!DIR_PART!" 2>nul
    if not errorlevel 1 (
        set "INPUT_PATH=!CD!\!BASE_PART!"
        popd
        echo 상대 경로가 다음 절대 경로로 변환되었습니다: !INPUT_PATH!
    ) else (
        echo 오류: 디렉토리 생성 또는 접근에 실패했습니다.
        exit /b 1
    )
) else if "!INPUT_PATH:~0,3!"=="../" (
    :: 상위 디렉토리 처리도 동일하게
    for %%i in ("!INPUT_PATH!") do (
        set "DIR_PART=%%~dpi"
        set "BASE_PART=%%~nxi"
    )
    
    mkdir "!DIR_PART!" 2>nul
    
    pushd "!DIR_PART!" 2>nul
    if not errorlevel 1 (
        set "INPUT_PATH=!CD!\!BASE_PART!"
        popd
        echo 상대 경로가 다음 절대 경로로 변환되었습니다: !INPUT_PATH!
    ) else (
        echo 오류: 디렉토리 생성 또는 접근에 실패했습니다.
        exit /b 1
    )
)

:: 설정 저장
echo { > "%CONFIG_FILE%"
echo     "milvus_data_path": "%INPUT_PATH%", >> "%CONFIG_FILE%"
echo     "created_at": "%date% %time%", >> "%CONFIG_FILE%"
echo     "last_modified": "%date% %time%" >> "%CONFIG_FILE%"
echo } >> "%CONFIG_FILE%"

echo 설정이 저장되었습니다: %CONFIG_FILE%

:: WSL 또는 VM 내부 볼륨 디렉토리 생성 안내
echo VM 또는 WSL에 데이터 디렉토리 생성을 확인하세요...
echo WSL 환경을 사용하는 경우 다음 명령을 WSL 터미널에서 실행하세요:
echo wsl -d Ubuntu sudo mkdir -p "%INPUT_PATH%/{etcd,minio,milvus,logs/{etcd,minio,milvus}}"
echo wsl -d Ubuntu sudo chown -R $(whoami):$(whoami) "%INPUT_PATH%"
echo wsl -d Ubuntu chmod -R 700 "%INPUT_PATH%/etcd"

:: 로컬 볼륨 디렉토리 생성 (설정 파일과 로그용)
mkdir volumes\logs\nginx 2>nul
mkdir volumes\logs\rag 2>nul
mkdir volumes\logs\reranker 2>nul
mkdir volumes\logs\prompt 2>nul

:: 권한 설정 검증
echo 내부 볼륨 권한 설정 확인 중...
echo etcd 디렉토리 권한:
dir "%INPUT_PATH%\etcd" 2>nul

echo 소유권 확인:
dir "%INPUT_PATH%" 2>nul

:: nginx 설정 파일 관리
:setup_nginx
set "mode=%~1"
echo nginx 설정 파일 설정 중 (%mode%)...

:: locations-enabled 디렉토리 확인
mkdir nginx\locations-enabled 2>nul
del /q nginx\locations-enabled\*.conf 2>nul

:: 모드에 따른 설정 파일 복사
if "%mode%"=="all" (
    :: 모두 복사
    copy /y nginx\templates\rag.conf.template nginx\locations-enabled\rag.conf >nul
    copy /y nginx\templates\reranker.conf.template nginx\locations-enabled\reranker.conf >nul
    copy /y nginx\templates\prompt.conf.template nginx\locations-enabled\prompt.conf >nul
) else if "%mode%"=="rag" (
    :: rag만 복사
    copy /y nginx\templates\rag.conf.template nginx\locations-enabled\rag.conf >nul
) else if "%mode%"=="reranker" (
    :: reranker만 복사
    copy /y nginx\templates\reranker.conf.template nginx\locations-enabled\reranker.conf >nul
) else if "%mode%"=="prompt" (
    :: prompt만 복사
    copy /y nginx\templates\prompt.conf.template nginx\locations-enabled\prompt.conf >nul
) else if "%mode%"=="rag-reranker" (
    :: rag와 reranker만 복사
    copy /y nginx\templates\rag.conf.template nginx\locations-enabled\rag.conf >nul
    copy /y nginx\templates\reranker.conf.template nginx\locations-enabled\reranker.conf >nul
)

:: nginx 재시작
docker ps | findstr "milvus-nginx" >nul
if not errorlevel 1 (
    echo nginx 재시작 중...
    docker restart milvus-nginx
)
goto :eof

:setup_reranker
set mode=%1
echo Reranker 설정 구성 중... (모드: %mode%)

if not exist "flashrank\Config.py" (
    echo 경고: flashrank\Config.py 파일이 없습니다.
    exit /b 1
)

if "%mode%"=="gpu" (
    where nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo 경고: NVIDIA 드라이버가 설치되어 있지 않습니다.
        echo GPU 모드를 사용하려면 NVIDIA 드라이버가 필요합니다.
        exit /b 1
    )
    copy /y reranker\requirements.gpu.txt reranker\requirements.txt
    copy /y reranker\Dockerfile.gpu reranker\Dockerfile
    powershell -Command "(Get-Content flashrank\Config.py) -replace 'torch_dtype=torch.float32', 'torch_dtype=torch.float16\n            device_map=\"auto\"' | Set-Content flashrank\Config.py"
) else (
    copy /y reranker\requirements.cpu.txt reranker\requirements.txt
    copy /y reranker\Dockerfile.cpu reranker\Dockerfile
    powershell -Command "(Get-Content flashrank\Config.py) -replace 'torch_dtype=torch.float16\n            device_map=\"auto\"', 'torch_dtype=torch.float32' | Set-Content flashrank\Config.py"
)
echo Reranker 설정이 성공적으로 변경되었습니다.
goto :eof

:: 서비스 시작
if "%1"=="all" (
    echo 모든 서비스 시작 중... (RAG + Reranker + Prompt + Ollama(CPU) + DB)
    call :setup_nginx all
    call :setup_reranker cpu
    docker compose --profile all --profile cpu-only up -d
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
) else if "%1"=="all-gpu" (
    echo 모든 서비스 시작 중... (RAG + Reranker + Prompt + Ollama(GPU) + DB)
    call :setup_nginx all
    call :setup_reranker gpu
    docker compose --profile all --profile gpu-only up -d
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
) else if "%1"=="rag" (
    echo RAG 서비스 시작 중... (RAG + DB)
    call :setup_nginx rag
    docker compose --profile rag-only up -d
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
) else if "%1"=="reranker" (
    echo Reranker 서비스만 시작
    call :setup_reranker cpu
    docker compose --profile reranker-only up -d
) else if "%1"=="reranker-gpu" (
    echo Reranker 서비스만 시작 (GPU 모드)
    call :setup_reranker gpu
    docker compose --profile reranker-only up -d
) else if "%1"=="prompt" (
    echo Prompt 서비스만 시작 중...
    call :setup_nginx prompt
    docker compose --profile prompt-only up -d
) else if "%1"=="rag-reranker" (
    echo RAG + Reranker 서비스 시작 중... (CPU 모드, DB 포함)
    call :setup_nginx rag-reranker
    call :setup_reranker cpu
    docker compose up -d nginx rag reranker standalone etcd etcd_init minio
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
) else if "%1"=="rag-reranker-gpu" (
    echo RAG + Reranker 서비스 시작 중... (GPU 모드, DB 포함)
    call :setup_nginx rag-reranker
    call :setup_reranker gpu
    docker compose up -d nginx rag reranker standalone etcd etcd_init minio
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
    docker restart milvus-standalone milvus-rag
) else if "%1"=="db" (
    echo 데이터베이스 서비스만 시작 중...
    docker compose --profile db-only up -d
) else if "%1"=="app-only" (
    echo 앱 서비스만 시작 중... (RAG + Reranker + Prompt + Ollama(CPU), DB 제외)
    call :setup_nginx all
    docker compose up -d nginx rag reranker prompt ollama
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
) else if "%1"=="app-only-gpu" (
    echo 앱 서비스만 시작 중... (RAG + Reranker + Prompt + Ollama(GPU), DB 제외)
    call :setup_nginx all
    call :setup_reranker gpu
    docker compose up -d nginx rag reranker prompt ollama-gpu
    docker exec -it milvus-rag pip uninstall numpy -y
    docker exec -it milvus-rag pip install numpy==1.24.4
) else if "%1"=="ollama" (
    echo Ollama 서비스 시작 중... (CPU 모드)
    docker compose --profile ollama-only --profile cpu-only up -d
    REM 모델 다운로드 스크립트 실행
    echo Ollama 모델 다운로드 중...
    REM 컨테이너가 완전히 시작될 때까지 잠시 대기
    timeout /t 3 /nobreak >nul
    REM 컨테이너가 실행 중인지 확인하고 모델 다운로드 수행
    docker exec milvus-ollama-cpu /app/init.sh
    if errorlevel 1 (
        echo 모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요.
        echo 컨테이너 로그 확인: docker logs milvus-ollama-cpu
    )
) else if "%1"=="ollama-gpu" (
    echo Ollama 서비스 시작 중... (GPU 모드)
    docker compose --profile gpu-only up -d
    REM 모델 다운로드 스크립트 실행
    echo Ollama 모델 다운로드 중...
    REM 컨테이너가 완전히 시작될 때까지 잠시 대기
    timeout /t 3 /nobreak >nul
    REM 컨테이너가 실행 중인지 확인하고 모델 다운로드 수행
    docker exec milvus-ollama-gpu /app/init.sh
    if errorlevel 1 (
        echo 모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요.
        echo 컨테이너 로그 확인: docker logs milvus-ollama-gpu
    )
) else if "%1"=="prompt_ollama" (
    echo Prompt와 Ollama 서비스 조합 시작 중... (CPU 모드)
    call :setup_nginx prompt
    docker compose --profile prompt-only --profile ollama-only --profile cpu-only up -d
    REM 모델 다운로드 스크립트 실행
    echo Ollama 모델 다운로드 중...
    REM 컨테이너가 완전히 시작될 때까지 잠시 대기
    timeout /t 3 /nobreak >nul
    REM 컨테이너가 실행 중인지 확인하고 모델 다운로드 수행
    docker exec milvus-ollama-cpu /app/init.sh
    if errorlevel 1 (
        echo 모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요.
        echo 컨테이너 로그 확인: docker logs milvus-ollama-cpu
    )
) else if "%1"=="prompt_ollama-gpu" (
    echo Prompt와 Ollama 서비스 조합 시작 중... (GPU 모드)
    call :setup_nginx prompt
    docker compose --profile prompt-only --profile gpu-only up -d
    REM 모델 다운로드 스크립트 실행
    echo Ollama 모델 다운로드 중...
    REM 컨테이너가 완전히 시작될 때까지 잠시 대기
    timeout /t 3 /nobreak >nul
    REM 컨테이너가 실행 중인지 확인하고 모델 다운로드 수행
    docker exec milvus-ollama-gpu /app/init.sh
    if errorlevel 1 (
        echo 모델 다운로드에 실패했습니다. Ollama API를 통해 수동으로 모델을 다운로드하세요.
        echo 컨테이너 로그 확인: docker logs milvus-ollama-gpu
    )
) else (
    echo Usage: %0 {all^|all-gpu^|rag^|reranker^|reranker-gpu^|prompt^|rag-reranker^|rag-reranker-gpu^|db^|app-only^|app-only-gpu^|ollama^|ollama-gpu^|prompt_ollama^|prompt_ollama-gpu}
    echo   all              - 모든 서비스 시작 (RAG + Reranker + Prompt + Ollama(CPU) + DB)
    echo   all-gpu          - 모든 서비스 시작 (RAG + Reranker + Prompt + Ollama(GPU) + DB)
    echo   rag              - RAG 서비스만 시작 (DB 포함)
    echo   reranker         - Reranker 서비스만 시작 (CPU 모드)
    echo   reranker-gpu     - Reranker 서비스만 시작 (GPU 모드)
    echo   prompt           - Prompt 서비스만 시작
    echo   rag-reranker     - RAG와 Reranker 서비스 시작 (CPU 모드, DB 포함)
    echo   rag-reranker-gpu - RAG와 Reranker 서비스 시작 (GPU 모드, DB 포함)
    echo   db               - 데이터베이스 서비스만 시작 (Milvus, Etcd, MinIO)
    echo   app-only         - 앱 서비스만 시작 (RAG + Reranker + Prompt + Ollama(CPU), DB 제외)
    echo   app-only-gpu     - 앱 서비스만 시작 (RAG + Reranker + Prompt + Ollama(GPU), DB 제외)
    echo   ollama           - Ollama 서비스만 시작 (CPU 모드)
    echo   ollama-gpu       - Ollama 서비스만 시작 (GPU 모드)
    echo   prompt_ollama    - Prompt와 Ollama 서비스 조합 (CPU 모드)
    echo   prompt_ollama-gpu - Prompt와 Ollama 서비스 조합 (GPU 모드)
    echo                 DB가 이미 실행 중일 때 코드 변경 후 사용
    exit /b 1
)

:: 서비스 상태 확인
echo 서비스 상태 확인 중...
docker ps | findstr "milvus api-gateway unified-nginx"

echo === 셋업 완료 ===
echo 시스템이 가동되었습니다. 다음 URL로 접근할 수 있습니다:

if "%1"=="all" (
    echo - RAG 서비스: http://localhost/rag/
    echo - Reranker 서비스: http://localhost/reranker/
    echo - 프롬프트 서비스: http://localhost/prompt/
    echo - 요약 API: http://localhost/prompt/summarize
    echo - Ollama API (CPU 모드): http://localhost:11434
    echo - Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)
) else if "%1"=="all-gpu" (
    echo - RAG 서비스: http://localhost/rag/
    echo - Reranker 서비스: http://localhost/reranker/
    echo - 프롬프트 서비스: http://localhost/prompt/
    echo - 요약 API: http://localhost/prompt/summarize
    echo - Ollama API (GPU 모드): http://localhost:11434
    echo - Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)
    echo - mistral, llama3 등 더 큰 모델을 사용할 수 있습니다.
) else if "%1"=="rag" (
    echo - RAG 서비스: http://localhost/rag/
    echo - Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)
) else if "%1"=="reranker" (
    echo - Reranker 서비스: http://localhost/reranker/
) else if "%1"=="prompt" (
    echo - 프롬프트 서비스: http://localhost/prompt/
    echo - 요약 API: http://localhost/prompt/summarize
) else if "%1"=="rag-reranker" (
    echo - RAG 서비스: http://localhost/rag/
    echo - Reranker 서비스: http://localhost/reranker/
    echo - Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)
) else if "%1"=="db" (
    echo - 데이터베이스 서비스만 시작되었습니다. 애플리케이션 서비스는 시작되지 않았습니다.
    echo - Milvus UI: http://localhost:9001 (사용자: minioadmin, 비밀번호: minioadmin)
) else if "%1"=="app-only" (
    echo - RAG 서비스: http://localhost/rag/
    echo - Reranker 서비스: http://localhost/reranker/
    echo - 프롬프트 서비스: http://localhost/prompt/
    echo - 요약 API: http://localhost/prompt/summarize
    echo - Ollama API (CPU 모드): http://localhost:11434
) else if "%1"=="app-only-gpu" (
    echo - RAG 서비스: http://localhost/rag/
    echo - Reranker 서비스: http://localhost/reranker/
    echo - 프롬프트 서비스: http://localhost/prompt/
    echo - 요약 API: http://localhost/prompt/summarize
    echo - Ollama API (GPU 모드): http://localhost:11434
    echo - mistral, llama3 등 더 큰 모델을 사용할 수 있습니다.
) else if "%1"=="ollama" (
    echo - Ollama API (CPU 모드): http://localhost:11434
    echo - Ollama 사용 예시: curl http://localhost:11434/api/tags
    echo - 참고: CPU 모드에서는 gemma:2b와 같은 작은 모델만 사용하세요.
) else if "%1"=="ollama-gpu" (
    echo - Ollama API (GPU 모드): http://localhost:11434
    echo - Ollama 사용 예시: curl http://localhost:11434/api/tags
    echo - mistral, llama3 등 더 큰 모델을 사용할 수 있습니다.
) else if "%1"=="prompt_ollama" (
    echo - 프롬프트 서비스: http://localhost/prompt/
    echo - 요약 API: http://localhost/prompt/summarize
    echo - 챗봇 API: http://localhost/prompt/chat
    echo - Ollama API (CPU 모드): http://localhost:11434
    echo - 참고: CPU 모드에서는 gemma:2b와 같은 작은 모델만 사용하세요.
) else if "%1"=="prompt_ollama-gpu" (
    echo - 프롬프트 서비스: http://localhost/prompt/
    echo - 요약 API: http://localhost/prompt/summarize
    echo - 챗봇 API: http://localhost/prompt/chat
    echo - Ollama API (GPU 모드): http://localhost:11434
    echo - mistral, llama3 등 더 큰 모델을 사용할 수 있습니다.
) 