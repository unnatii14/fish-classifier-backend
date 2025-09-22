@echo off
echo ===========================================
echo Fish Classifier API - Complete Test Suite
echo ===========================================
echo.

cd /d "c:\Users\Hetvi\Downloads\fish-classifier-backend-main (3)\fish-classifier-backend-main"

echo Step 1: Starting API Server...
echo.
python start_api_server.py

echo.
echo Step 2: Testing API (with server check)...
echo.
python test_with_server_check.py

echo.
echo Step 3: Opening Web Interface...
echo.
start simple_test_interface.html

echo.
echo ===========================================
echo Testing Complete!
echo ===========================================
echo.
pause