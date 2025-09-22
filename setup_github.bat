@echo off
echo ===========================================
echo Fish Classifier - GitHub Setup Automation
echo ===========================================
echo.

cd /d "c:\Users\Hetvi\Downloads\fish-classifier-backend-main (3)\fish-classifier-backend-main"

echo Step 1: Initializing Git Repository...
git init
echo.

echo Step 2: Adding files to Git...
git add .
echo.

echo Step 3: Creating initial commit...
git commit -m "Initial commit: Fish Classifier API with 31 species support and web interface"
echo.

echo Step 4: Repository Setup Instructions
echo ===========================================
echo.
echo NEXT STEPS (Manual):
echo.
echo 1. Go to GitHub.com and create a new repository
echo    - Name: fish-classifier-api
echo    - Description: AI-powered fish species classification API
echo    - Don't initialize with README
echo.
echo 2. Copy the repository URL (it will look like):
echo    https://github.com/YOURUSERNAME/fish-classifier-api.git
echo.
echo 3. Come back here and run these commands:
echo.
echo    git remote add origin YOUR_REPOSITORY_URL
echo    git branch -M main
echo    git push -u origin main
echo.
echo ===========================================

set /p repo_url="Enter your GitHub repository URL (or press Enter to skip): "

if not "%repo_url%"=="" (
    echo.
    echo Step 5: Connecting to GitHub...
    git remote add origin %repo_url%
    
    echo.
    echo Step 6: Setting main branch...
    git branch -M main
    
    echo.
    echo Step 7: Pushing to GitHub...
    echo You may be prompted for GitHub credentials...
    git push -u origin main
    
    echo.
    echo ===========================================
    echo SUCCESS! Your project is now on GitHub!
    echo ===========================================
    echo.
    echo Repository URL: %repo_url%
    echo.
    echo Your Fish Classifier API is now publicly available!
) else (
    echo.
    echo Skipped GitHub upload. Follow manual steps above.
)

echo.
echo ===========================================
echo Setup Complete!
echo ===========================================
echo.
echo What was uploaded:
echo ✅ API code (main.py)
echo ✅ Web interfaces (HTML files)
echo ✅ Test scripts
echo ✅ Documentation
echo ✅ Configuration files
echo.
echo What was NOT uploaded (due to size):
echo ❌ Model files (*.pth) 
echo ❌ Data files (*.npy)
echo ❌ Cache files
echo.
pause