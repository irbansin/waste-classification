#!/bin/bash
# Development startup script for Waste Classification Project
set -e

# 1. Create and activate virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment in .venv ..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Install dependencies
echo "[INFO] Installing requirements ..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Start FastAPI backend (in background)
echo "[INFO] Starting FastAPI backend at http://127.0.0.1:8001 ..."
nohup uvicorn src.api:app --reload --port 8001 > fastapi_8001.log 2>&1 &
FASTAPI_PID=$!
echo "[INFO] FastAPI PID: $FASTAPI_PID (log: fastapi_8001.log)"

# 4. Start Streamlit web UI (in foreground)
echo "[INFO] Starting Streamlit web UI ..."
echo "[INFO] Access at http://localhost:8501"
streamlit run src/webui.py

# 5. Cleanup FastAPI backend on exit
trap "echo '[INFO] Stopping FastAPI backend ...'; kill $FASTAPI_PID" EXIT
