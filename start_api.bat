@echo off
echo Starting Fish Classifier API...
cd "c:\Users\Hetvi\Downloads\fish-classifier-backend-main (3)\fish-classifier-backend-main"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause