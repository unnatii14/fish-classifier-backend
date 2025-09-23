Write-Host "🚀 Deploying Railway Health Check Fixes" -ForegroundColor Green
Write-Host "=" * 50

# Check git status
Write-Host "🔄 Checking git status..." -ForegroundColor Yellow
git status

# Add all files
Write-Host "🔄 Adding files..." -ForegroundColor Yellow
git add .

# Commit changes
Write-Host "🔄 Committing changes..." -ForegroundColor Yellow
git commit -m "Fix Railway deployment health checks and optimize for memory constraints"

# Push to Railway
Write-Host "🔄 Pushing to Railway..." -ForegroundColor Yellow
git push

Write-Host ""
Write-Host "=" * 50
Write-Host "✅ Railway fixes deployed!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Monitor Railway deployment logs"
Write-Host "2. Check health status at your-app.railway.app/health"
Write-Host "3. Test API at your-app.railway.app/"
Write-Host "4. Run: python test_railway_optimized.py https://your-app.railway.app"