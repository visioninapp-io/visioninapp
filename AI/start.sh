#!/bin/bash
# Start AI Message-driven Service

echo "Starting AI Service (Message-driven)..."
echo "======================================="

# ✅ 명시적으로 Windows Python 경로 지정
PYTHON="/c/Users/SSAFY/AppData/Local/Programs/Python/Python311/python.exe"

# ✅ 실제로 어떤 Python을 쓰는지 출력
echo "Using Python: $PYTHON"
"$PYTHON" --version

# ✅ CUDA 확인
"$PYTHON" -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "❌ Python/Torch not found. Please install dependencies:"
    echo "   $PYTHON -m pip install -r requirements.txt"
    exit 1
fi

# ✅ RabbitMQ 연결 확인 (선택사항)
echo "🐰 Checking RabbitMQ connection..."
"$PYTHON" -c "
import pika
import os
try:
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=os.getenv('RABBITMQ_HOST', 'localhost'),
        port=int(os.getenv('RABBITMQ_PORT', '5672')),
        credentials=pika.PlainCredentials(
            os.getenv('RABBITMQ_USER', 'guest'),
            os.getenv('RABBITMQ_PASSWORD', 'guest')
        )
    ))
    connection.close()
    print('✅ RabbitMQ connection successful')
except Exception as e:
    print(f'⚠️  RabbitMQ connection failed: {e}')
    print('   Service will still start but may not receive messages')
"

# ✅ 메시지 기반 서비스 시작
echo "🚀 Starting AI Message-driven Service..."
"$PYTHON" ai_service.py
