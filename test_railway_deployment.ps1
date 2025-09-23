# Railway Deployment Test Script
# Use this to verify your Railway deployment is working

$RAILWAY_URL = "https://lively-amazement-production-e614.up.railway.app"

Write-Host "🔍 Testing Railway Deployment..." -ForegroundColor Green
Write-Host "URL: $RAILWAY_URL" -ForegroundColor Cyan
Write-Host ""

# Test 1: Root endpoint
Write-Host "1️⃣ Testing Root Endpoint (/)" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$RAILWAY_URL/" -Method GET -ErrorAction Stop
    Write-Host "✅ Root endpoint working!" -ForegroundColor Green
    Write-Host "Response: $($response | ConvertTo-Json -Depth 2)" -ForegroundColor White
} catch {
    Write-Host "❌ Root endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 2: Health endpoint
Write-Host "2️⃣ Testing Health Endpoint (/health)" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$RAILWAY_URL/health" -Method GET -ErrorAction Stop
    Write-Host "✅ Health endpoint working!" -ForegroundColor Green
    Write-Host "Response: $($response | ConvertTo-Json -Depth 2)" -ForegroundColor White
} catch {
    Write-Host "❌ Health endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 3: Info endpoint
Write-Host "3️⃣ Testing Info Endpoint (/info)" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$RAILWAY_URL/info" -Method GET -ErrorAction Stop
    Write-Host "✅ Info endpoint working!" -ForegroundColor Green
    Write-Host "Response: $($response | ConvertTo-Json -Depth 2)" -ForegroundColor White
} catch {
    Write-Host "❌ Info endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 4: Classes endpoint
Write-Host "4️⃣ Testing Classes Endpoint (/classes)" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$RAILWAY_URL/classes" -Method GET -ErrorAction Stop
    Write-Host "✅ Classes endpoint working!" -ForegroundColor Green
    Write-Host "Total classes: $($response.total)" -ForegroundColor White
} catch {
    Write-Host "❌ Classes endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 5: Predict endpoint (demo)
Write-Host "5️⃣ Testing Predict Endpoint (/predict)" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$RAILWAY_URL/predict" -Method POST -ErrorAction Stop
    Write-Host "✅ Predict endpoint working!" -ForegroundColor Green
    Write-Host "Demo prediction: $($response.demo_predictions[0].species)" -ForegroundColor White
} catch {
    Write-Host "❌ Predict endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# Test 6: FastAPI docs
Write-Host "6️⃣ Testing API Documentation (/docs)" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$RAILWAY_URL/docs" -Method GET -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ API docs accessible!" -ForegroundColor Green
        Write-Host "Status: $($response.StatusCode)" -ForegroundColor White
    }
} catch {
    Write-Host "❌ API docs failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎯 Test Results Summary:" -ForegroundColor Magenta
Write-Host "If all tests pass ✅, your Railway deployment is fully functional!" -ForegroundColor Green
Write-Host "If any tests fail ❌, check Railway logs and redeploy." -ForegroundColor Yellow
Write-Host ""
Write-Host "📱 For Flutter Integration:" -ForegroundColor Cyan
Write-Host "const String API_BASE_URL = `"$RAILWAY_URL`";" -ForegroundColor White