# Simple Railway Deployment Test
# Copy and paste these commands one by one

# Test 1: Basic connectivity
Invoke-WebRequest -Uri "https://lively-amazement-production-e614.up.railway.app/" -UseBasicParsing

# Test 2: Health check
Invoke-RestMethod -Uri "https://lively-amazement-production-e614.up.railway.app/health"

# Test 3: API info
Invoke-RestMethod -Uri "https://lively-amazement-production-e614.up.railway.app/info"

# Test 4: Check API documentation
Start-Process "https://lively-amazement-production-e614.up.railway.app/docs"

# SUCCESS INDICATORS:
# ✅ Test 1: Should return HTTP 200 status
# ✅ Test 2: Should return {"status": "healthy"}  
# ✅ Test 3: Should return API information JSON
# ✅ Test 4: Should open FastAPI documentation page