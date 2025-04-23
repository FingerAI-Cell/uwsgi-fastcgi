from flask import Flask, request, jsonify
import os
import logging
import json
import requests
from typing import Optional, Dict, Any

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(
    filename='/var/log/vision/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 설정 파일 로드
def load_config():
    try:
        with open('/vision/config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {str(e)}")
        return {}

config = load_config()

def call_ollama_api(model: str, prompt: str, image_url: str) -> Optional[Dict[str, Any]]:
    """Ollama API를 호출하여 이미지 분석을 수행합니다."""
    try:
        ollama_url = os.getenv('OLLAMA_ENDPOINT', config.get('ollama_endpoint', 'http://ollama:11434'))
        
        # API 요청 데이터
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_url],
            "stream": False  # 스트리밍 비활성화
        }
        
        # API 호출
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=30  # 30초 타임아웃 설정
        )
        
        # 응답 검증
        response.raise_for_status()
        result = response.json()
        
        if not result.get('response'):
            logger.warning(f"Ollama API 응답에 'response' 필드가 없습니다: {result}")
            return None
            
        return result
        
    except requests.exceptions.Timeout:
        logger.error("Ollama API 호출 시간 초과")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API 호출 실패: {str(e)}")
        return None
    except json.JSONDecodeError:
        logger.error("Ollama API 응답을 JSON으로 파싱할 수 없습니다")
        return None
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """서비스 상태를 확인합니다."""
    return jsonify({
        "status": "healthy",
        "service": "vision",
        "default_model": config.get('default_model')
    }), 200

@app.route('/analyze', methods=['POST'])
def analyze_media():
    """이미지 분석을 수행합니다."""
    try:
        # 요청 데이터 검증
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON 데이터가 필요합니다"}), 400
            
        if 'url' not in data:
            return jsonify({"error": "이미지 URL이 필요합니다"}), 400
            
        # 파라미터 추출
        image_url = data['url']
        prompt = data.get('prompt', "이 이미지에 대해 설명해주세요")
        model = data.get('model', config.get('default_model', 'llama3.2-vision'))
        
        # Ollama API 호출
        result = call_ollama_api(model, prompt, image_url)
        if not result:
            return jsonify({"error": "이미지 분석에 실패했습니다"}), 500
            
        # 응답 반환
        return jsonify({
            "description": result.get('response'),
            "image_url": image_url,
            "model": model,
            "total_duration": result.get('total_duration'),
            "load_duration": result.get('load_duration'),
            "prompt_eval_count": result.get('prompt_eval_count'),
            "eval_count": result.get('eval_count'),
            "eval_duration": result.get('eval_duration')
        }), 200
            
    except Exception as e:
        logger.error(f"미디어 분석 중 오류 발생: {str(e)}")
        return jsonify({"error": "미디어 분석 중 오류가 발생했습니다"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 