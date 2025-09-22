# Fish Classifier API Testing Guide

## How to Test Your API is Working Correctly

### Step 1: Start Your API Server

```bash
# Navigate to your API directory
cd "c:\Users\Hetvi\Downloads\fish-classifier-backend-main (3)\fish-classifier-backend-main"

# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will start at: `http://localhost:8000`

### Step 2: Quick Test Methods

#### Method A: Simple Browser Test
1. Open browser and go to: `http://localhost:8000`
2. You should see API info with `"model_loaded": true`
3. Go to: `http://localhost:8000/health` to check health status

#### Method B: Quick Python Test
```bash
# Run the quick test script
python test_api_quick.py

# Or test with a real fish image
python test_api_quick.py your_fish_image.jpg
```

#### Method C: Comprehensive Test Suite
```bash
# Run full test suite
python test_api_comprehensive.py
```

### Step 3: Manual Testing with curl/PowerShell

#### Test Health Endpoint
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

#### Test Classes Endpoint
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/classes" -Method Get
```

#### Test Prediction with Base64
```powershell
# Create test payload (you'll need to replace with actual base64 image)
$payload = @{
    image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict-base64" -Method Post -Body $payload -ContentType "application/json"
```

### Step 4: Verify Correct Fish Classification

#### What to Look For:

1. **API Status**:
   - `"model_loaded": true`
   - `"status": "healthy"`
   - `"success": true` in prediction responses

2. **Correct Predictions**:
   - Species names from the 31-class list
   - Confidence scores between 0-100%
   - Top-5 predictions that make sense

3. **Expected Fish Species**:
   ```
   'Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch',
   'Fourfinger Threadfin', 'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish',
   'Gourami', 'Grass Carp', 'Green Spotted Puffer', 'Indian Carp', 'Indo-Pacific Tarpon',
   'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish', 'Mosquito Fish',
   'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
   'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia'
   ```

### Step 5: Test with Your Flutter App

#### API Endpoints for Flutter:

1. **File Upload Prediction**:
   ```
   POST http://localhost:8000/predict
   Content-Type: multipart/form-data
   Body: file parameter with image
   ```

2. **Base64 Prediction** (recommended for mobile):
   ```
   POST http://localhost:8000/predict-base64
   Content-Type: application/json
   Body: {"image": "base64_encoded_image_string"}
   ```

3. **Similarity Search**:
   ```
   POST http://localhost:8000/find-similar
   Content-Type: multipart/form-data
   Body: file parameter with image, top_k parameter (optional)
   ```

### Step 6: Troubleshooting

#### If API shows "demo_mode": true
1. Check if `best_model_efficientnet.pth` exists in the API directory
2. Verify file size (should be ~20MB)
3. Check API logs for loading errors

#### If predictions are wrong
1. Compare with Streamlit demo results
2. Check if model loading shows "fixed approach" message
3. Verify image preprocessing (224x224, RGB format)

#### If API won't start
1. Check if port 8000 is available
2. Install requirements: `pip install -r requirements.txt`
3. Check Python/PyTorch installation

### Expected Good Results:

```json
{
  "success": true,
  "predictions": [
    {"species": "Bangus", "confidence": 94.2},
    {"species": "Tilapia", "confidence": 3.1},
    {"species": "Catfish", "confidence": 1.8}
  ],
  "top_prediction": "Bangus",
  "confidence": 94.2
}
```

### Integration with Flutter

Once verified working, update your Flutter app to use:
- **Production URL**: Replace `localhost:8000` with your deployed URL
- **Base64 method**: More reliable for mobile file uploads
- **Error handling**: Check for `"success": true` in responses

Your API should now give the same accurate fish species classifications as your working Streamlit demo! 🐠✅