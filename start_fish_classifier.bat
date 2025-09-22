@echo off
echo ========================================
echo   🐠 Fish Classifier - Complete Setup
echo ========================================
echo.

cd /d "%~dp0"

echo 🔧 Starting Fish Classifier API...
echo.
start "Fish Classifier API" cmd /k "python -m uvicorn main:app --host 127.0.0.1 --port 8000 && pause"

echo ⏳ Waiting for API to start...
timeout /t 5 /nobreak >nul

echo.
echo 🌐 Opening Web Test Interface...
start "" "simple_test_interface.html"

echo.
echo ========================================
echo ✅ Setup Complete!
echo ========================================
echo.
echo 📍 API Running at: http://127.0.0.1:8000
echo 🌐 Web Interface opened in your browser
echo.
echo 💡 You can also manually open:
echo    - simple_test_interface.html
echo    - fish_classifier_web_interface.html
echo.
echo ⚠️  Keep this window open to maintain the API
echo    Close this window to stop the API
echo.
pause