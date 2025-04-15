#!/bin/bash

# 로그 시작
echo "Ollama 모델 다운로드 스크립트"

# GPU 감지
if nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU가 감지되었습니다. GPU 모드로 실행합니다."
    GPU_AVAILABLE=true
else
    echo "GPU가 감지되지 않았습니다. CPU 모드로 실행합니다."
    GPU_AVAILABLE=false
fi

# 사용 가능한 RAM 체크 (GB 단위)
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "사용 가능한 RAM: ${TOTAL_RAM_GB}GB"

# 모델 목록 파일 체크
if [ ! -f "/app/models.txt" ]; then
    echo "모델 목록 파일이 없습니다. 기본 모델만 설치합니다."
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "GPU 모드: mistral 모델 다운로드 중..."
        ollama pull mistral
    else
        if [ "$TOTAL_RAM_GB" -ge 13 ]; then
            echo "CPU 모드 (13GB+ RAM): mistral 모델 다운로드 중..."
            ollama pull mistral
        else
            echo "CPU 모드 (제한된 RAM): gemma:2b 모델 다운로드 중..."
            ollama pull gemma:2b
        fi
    fi
    echo "기본 모델 다운로드 완료!"
    exit 0
fi

# 모델 목록 파일에서 모델 가져오기
while IFS= read -r line || [ -n "$line" ]; do
    # Windows의 줄 끝 문자(\r) 제거
    line=$(echo "$line" | tr -d '\r')
    
    # 빈 줄이나 주석 건너뛰기
    if [ -z "$line" ] || [[ "$line" == \#* ]]; then
        continue
    fi
    
    # CPU 모드에서 RAM 체크 및 모델 선택
    if [ "$GPU_AVAILABLE" = false ]; then
        if [ "$TOTAL_RAM_GB" -lt 13 ] && [[ "$line" != gemma:2b ]]; then
            echo "RAM 부족 (${TOTAL_RAM_GB}GB): 무거운 모델($line)은 건너뜁니다."
            continue
        elif [ "$TOTAL_RAM_GB" -lt 24 ] && [[ "$line" == mixtral* ]]; then
            echo "RAM 부족 (${TOTAL_RAM_GB}GB): 대형 모델($line)은 건너뜁니다."
            continue
        fi
    fi
    
    echo "모델 다운로드 중: $line"
    ollama pull "$line"
done < "/app/models.txt"

echo "Ollama 모델 다운로드 완료!" 