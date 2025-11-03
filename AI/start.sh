#!/bin/bash
# Start AI Service on GPU server

echo "Starting AI Service..."
echo "====================="

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

# ✅ 포트 8001 사용 중인지 확인
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8001 is already in use!"
    echo "Stop the existing service first:"
    echo "   kill $(lsof -t -i:8001)"
    exit 1
fi

# ✅ 서비스 시작
echo "🚀 Starting AI Service on port 8001..."
"$PYTHON" -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
