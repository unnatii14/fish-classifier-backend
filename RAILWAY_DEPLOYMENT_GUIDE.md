# Railway Deployment Guide for Fish Classifier API

This guide will help you deploy your Fish Classifier API to Railway without any issues.

## Pre-Deployment Fixes Applied ✅

Your project has been optimized for Railway deployment with the following fixes:

### 1. **Port Configuration** ✅
- Fixed `main.py` to use Railway's `$PORT` environment variable
- Added fallback to port 8000 for local development

### 2. **Memory Optimization** ✅
- Added Railway-specific memory optimizations
- Limited PyTorch threads for Railway's resource constraints
- Disabled CUDA support (CPU-only deployment)
- Added memory mapping for model loading

### 3. **Dependency Management** ✅
- Updated `requirements.txt` with Railway-compatible versions
- Added `gunicorn` for production WSGI server
- Specified PyTorch CPU-only versions to reduce build size

### 4. **Railway Configuration** ✅
- Updated `railway.toml` with optimized settings
- Set restart policy to "on_failure" instead of "never"
- Added memory optimization environment variables
- Updated PyTorch version to match requirements

### 5. **Health Checks** ✅
- Added `/health` endpoint for Railway monitoring
- Enhanced root endpoint with deployment status
- Added `/info` endpoint for API documentation

### 6. **File Size Monitoring** ✅
- Added model file size checking in `load_model()`
- Warning system for large files (>500MB Railway limit)
- Memory-efficient model loading

## Railway Deployment Steps

### Step 1: Push Updated Code to GitHub
```bash
cd "c:\Users\Hetvi\Downloads\fish-classifier-backend-main (3)\fish-classifier-backend-main"

git add .
git commit -m "Railway deployment optimization: 
- Add PORT environment variable support
- Memory optimization for Railway
- Enhanced health checks
- Updated dependencies for Railway compatibility"

git push origin main
```

### Step 2: Deploy to Railway

#### Option A: Railway CLI (Recommended)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Create new project and deploy
railway new
railway up
```

#### Option B: Railway Dashboard
1. Go to https://railway.app
2. Sign up/Login with your GitHub account
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your `fish-classifier-backend` repository
6. Railway will automatically detect Python and use your configurations

### Step 3: Configure Environment Variables (Optional)
In Railway dashboard, add these if needed:
- `PYTHON_VERSION=3.10`
- `PYTHONUNBUFFERED=1`
- `PYTHONDONTWRITEBYTECODE=1`

### Step 4: Monitor Deployment
1. Check deployment logs in Railway dashboard
2. Wait for build to complete (may take 5-10 minutes)
3. Test your deployed API endpoints

## Expected Deployment Behavior

### ✅ What Should Work:
- **Model Loading**: Optimized for Railway's memory constraints
- **Health Checks**: `/health` endpoint for monitoring
- **API Endpoints**: All classification and similarity endpoints
- **CORS**: Enabled for web/mobile app integration
- **Auto-restart**: On failure (not on crash)

### ⚠️ Potential Issues & Solutions:

#### Issue 1: Model File Too Large
**Symptoms**: Build fails with file size error
**Solution**: Model files >100MB need Git LFS
```bash
git lfs track "*.pth"
git lfs track "*.npy"
git add .gitattributes
git add best_model_efficientnet.pth val_embeddings.npy
git commit -m "Add large files with Git LFS"
git push origin main
```

#### Issue 2: Memory Limit Exceeded
**Symptoms**: Deployment crashes with memory error
**Solution**: Added memory optimizations prevent this

#### Issue 3: Slow Cold Starts
**Symptoms**: First request takes long time
**Solution**: Health check endpoint keeps service warm

#### Issue 4: Build Timeout
**Symptoms**: Build takes >30 minutes
**Solution**: Optimized PyTorch installation prevents this

## Testing Your Deployed API

### 1. Basic Health Check
```bash
curl https://your-app.railway.app/health
```

### 2. API Information
```bash
curl https://your-app.railway.app/info
```

### 3. Fish Classification Test
```bash
curl -X POST "https://your-app.railway.app/predict-base64" \
  -H "Content-Type: application/json" \
  -d '{"image":"your_base64_image","top_k":3}'
```

### 4. Update Your Flutter App
Replace localhost URLs with your Railway URL:
```dart
const String API_BASE_URL = "https://your-app.railway.app";
```

## Railway URLs

After successful deployment:
- **Main URL**: `https://your-project-name.railway.app`
- **Health Check**: `https://your-project-name.railway.app/health`
- **API Docs**: `https://your-project-name.railway.app/docs`

## Performance Expectations

### Railway Free Tier:
- **Memory**: 512MB RAM
- **CPU**: Shared vCPU
- **Storage**: 1GB
- **Bandwidth**: 100GB/month
- **Uptime**: May sleep after inactivity

### Railway Pro Tier:
- **Memory**: Up to 8GB RAM
- **CPU**: Dedicated vCPU
- **Storage**: 100GB
- **Bandwidth**: 1TB/month
- **Uptime**: Always on

## Deployment Checklist ✅

Before deploying, ensure:
- [ ] All files committed to GitHub
- [ ] Model files under 500MB (or using Git LFS)
- [ ] Railway account created
- [ ] GitHub connected to Railway
- [ ] Railway CLI installed (optional)

## Troubleshooting

### Build Fails
1. Check Railway logs for specific error
2. Verify all files are committed to GitHub
3. Check if model files need Git LFS

### App Crashes
1. Check `/health` endpoint
2. Review Railway application logs
3. Monitor memory usage in Railway dashboard

### Slow Performance
1. Consider upgrading to Railway Pro
2. Optimize model loading (already done)
3. Use Railway's auto-scaling features

## Success Indicators

✅ **Successful Deployment Shows**:
- Green status in Railway dashboard
- `/health` returns 200 status
- `/` shows API status with `"deployment": "Railway optimized"`
- All endpoints respond correctly
- No memory or timeout errors in logs

Your project is now Railway-ready! 🚀