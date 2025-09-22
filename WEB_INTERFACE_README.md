# 🐠 Fish Classifier Web Interface

This folder contains several web interfaces to test your Fish Classifier API visually, similar to your Streamlit demo.

## 🚀 Quick Start

### Option 1: One-Click Setup
1. Double-click `start_fish_classifier.bat`
2. This will start your API and open the web interface automatically

### Option 2: Manual Setup
1. Start your API:
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
2. Open any of the HTML files in your browser:
   - `simple_test_interface.html` (Recommended - most complete)
   - `fish_classifier_web_interface.html` (Advanced version)

### Option 3: Flask Web Server
1. Install Flask: `pip install flask`
2. Start your API (port 8000)
3. Run: `python web_interface.py`
4. Open: http://127.0.0.1:5000

## 📁 Files Included

### Web Interfaces
- **`simple_test_interface.html`** ⭐ 
  - Complete, self-contained interface
  - No external dependencies
  - Drag & drop support
  - Real-time API testing
  - Mobile responsive

- **`fish_classifier_web_interface.html`**
  - Advanced interface with enhanced styling
  - Similar to Streamlit layout
  - Professional appearance

- **`web_interface.py`**
  - Flask-based web server
  - Server-side API integration
  - For production deployments

### Utilities
- **`start_fish_classifier.bat`**
  - One-click startup script
  - Starts API + opens web interface
  - Windows only

- **Test Scripts**
  - `test_api_quick.py` - Quick API validation
  - `test_api_comprehensive.py` - Full API testing

## 🎯 Features

### ✅ What the Web Interface Can Do:
- **Fish Species Classification**: Upload images and get AI predictions
- **Top-5 Predictions**: Shows confidence scores for multiple species
- **Similarity Search**: Find similar fish images from training data
- **Real-time API Testing**: Verify API connection and status
- **Drag & Drop Upload**: Easy image selection
- **Mobile Responsive**: Works on phones and tablets
- **Error Handling**: Clear error messages and troubleshooting

### 🐠 Supported Fish Species (31 total):
```
Bangus, Big Head Carp, Black Spotted Barb, Catfish, Climbing Perch,
Fourfinger Threadfin, Freshwater Eel, Glass Perchlet, Goby, Gold Fish,
Gourami, Grass Carp, Green Spotted Puffer, Indian Carp, Indo-Pacific Tarpon,
Jaguar Gapote, Janitor Fish, Knifefish, Long-Snouted Pipefish, Mosquito Fish,
Mudfish, Mullet, Pangasius, Perch, Scat Fish, Silver Barb, Silver Carp,
Silver Perch, Snakehead, Tenpounder, Tilapia
```

## 🔧 Configuration

### API URL Configuration
- Default: `http://127.0.0.1:8000`
- Change in the web interface or modify the HTML files
- For production: Update to your deployed API URL

### API Endpoints Used
- `GET /health` - Check API status
- `GET /classes` - Get available fish species
- `POST /predict-base64` - Image classification
- `POST /find-similar` - Similarity search

## 🧪 Testing Your API

### Step 1: Start Everything
```bash
# Option A: Use the batch file
start_fish_classifier.bat

# Option B: Manual
python -m uvicorn main:app --host 127.0.0.1 --port 8000
# Then open simple_test_interface.html
```

### Step 2: Test with Images
1. **Upload a fish image** using the web interface
2. **Check the predictions** - should show species names and confidence scores
3. **Verify similarity search** - should show related images from training data

### Step 3: Expected Results
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

## 🐛 Troubleshooting

### API Connection Issues
- ❌ **"API Connection Failed"**
  - Check if your API is running on port 8000
  - Verify the API URL in the interface
  - Try: `python test_api_quick.py`

### Model Loading Issues
- ❌ **"Model not loaded"**
  - Ensure `best_model_efficientnet.pth` exists
  - Check API logs for loading errors
  - Verify the model file isn't corrupted

### Prediction Issues
- ❌ **"Demo mode" responses**
  - Model file not found or corrupted
  - Check the model loading approach in main.py
  - Ensure you're using the fixed version

### Image Upload Issues
- ❌ **Upload fails**
  - Check image file format (JPG, PNG supported)
  - Verify image isn't too large (>10MB)
  - Try a different image

## 🌐 Production Deployment

### For Flutter App Integration
1. Deploy your API to a cloud service (Railway, Heroku, etc.)
2. Update the API URL in your Flutter app
3. Use the `/predict-base64` endpoint for mobile compatibility
4. Test with the web interface first to verify functionality

### For Web Deployment
1. Upload the HTML files to any web server
2. Update the API URL in the JavaScript
3. Ensure CORS is enabled in your FastAPI (already configured)

## 📊 Comparison with Streamlit

| Feature | Streamlit | Web Interface |
|---------|-----------|---------------|
| Fish Classification | ✅ | ✅ |
| Top-5 Predictions | ✅ | ✅ |
| Similarity Search | ✅ | ✅ |
| Image Upload | ✅ | ✅ (+ Drag & Drop) |
| Real-time Results | ✅ | ✅ |
| Mobile Support | ❌ | ✅ |
| API Testing | ❌ | ✅ |
| No Dependencies | ❌ | ✅ (HTML version) |

The web interface provides the same functionality as your Streamlit demo with additional features for testing and production use! 🎉