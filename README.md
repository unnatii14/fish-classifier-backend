# 🐟 Fish Classifier API

A production-ready FastAPI application for fish species classification and similarity search using deep learning. Deployed on Railway with EfficientNet-B0 architecture for accurate fish identification across 31 species.

![Fish Classification](https://img.shields.io/badge/Fish_Species-31-blue)
![Model](https://img.shields.io/badge/Model-EfficientNet--B0-green)
![Framework](https://img.shields.io/badge/Framework-FastAPI-red)
![Python](https://img.shields.io/badge/Python-3.10+-brightgreen)

## 🌐 Live Demo
**API URL**: https://web-production-cc66.up.railway.app  
**Interactive Docs**: https://web-production-cc66.up.railway.app/docs

## 🚀 Features
- **31 Fish Species Classification** - Accurate identification using EfficientNet-B0
- **Image Similarity Search** - Find visually similar fish using deep embeddings
- **RESTful API** - Well-documented FastAPI with automatic OpenAPI docs
- **Production Ready** - Optimized for Railway deployment with health checks
- **CORS Enabled** - Ready for web and mobile app integration
- **Web Interface** - Built-in testing interface for easy validation
- **Base64 Support** - Perfect for mobile applications

## 📱 Mobile App Ready
This API is specifically designed for Flutter and mobile app integration with base64 image support and CORS configuration.

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip package manager

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/fish-classifier-api.git
cd fish-classifier-api

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn main:app --host 127.0.0.1 --port 8001

# Test the API
python test_with_server_check.py
```

### Required Files
To run locally, you need these model files:
- `best_model_efficientnet.pth` - Trained model weights
- `val_embeddings.npy` - Pre-computed embeddings
- `val_image_paths.npy` - Dataset image paths

## 📚 API Endpoints

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| GET | `/` | API status and info | - |
| GET | `/health` | Health check | - |
| GET | `/classes` | List all 31 fish species | - |
| POST | `/predict` | Classify fish image | `file`: image |
| POST | `/predict-base64` | Classify base64 image | `image`: base64 string, `top_k`: int |
| POST | `/find-similar` | Find similar fish images | `file`: image, `top_k`: int |
| POST | `/find-similar-base64` | Find similar base64 images | `image`: base64 string, `top_k`: int |
| GET | `/fish-image/{index}` | Get fish image by index | `index`: int |

### Example Responses

**Classification Response:**
```json
{
  "predictions": [
    {
      "species": "Bangus",
      "confidence": 0.9894
    },
    {
      "species": "Indo-Pacific Tarpon", 
      "confidence": 0.0002
    }
  ]
}
```

**Similar Images Response:**
```json
{
  "similar_images": [
    {
      "species_name": "Bangus",
      "filename": "Bangus_101.jpg", 
      "similarity": 0.9234,
      "index": 42
    }
  ]
}
```

## 🧪 Testing

### Web Interface Testing
1. Start the server locally
2. Open `simple_test_interface.html` in your browser
3. Upload a fish image and test classification
4. Verify similar images functionality

### Automated Testing
```bash
# Complete API test
python test_with_server_check.py

# Quick endpoint test  
python quick_api_test.py

# One-click testing (Windows)
test_everything.bat
```

### API Dashboard
Open `api_test_dashboard.html` for a comprehensive testing interface with visual feedback.

## 📱 Flutter Integration

Perfect for mobile apps with base64 image support:

```dart
// Classification example
final response = await http.post(
  Uri.parse('$apiUrl/predict-base64'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'image': base64Image,
    'top_k': 3
  }),
);

// Similar images example  
final similarResponse = await http.post(
  Uri.parse('$apiUrl/find-similar-base64'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'image': base64Image,
    'top_k': 5
  }),
);
```

## Quick Start
```bash
# Health check
curl https://web-production-cc66.up.railway.app/health

# Get all fish species
curl https://web-production-cc66.up.railway.app/classes

# Classify fish (upload image)
curl -X POST "https://web-production-cc66.up.railway.app/predict" \
  -F "file=@your_fish_image.jpg"

# Find similar fish
curl -X POST "https://web-production-cc66.up.railway.app/find-similar" \
  -F "file=@your_fish_image.jpg" \
  -F "top_k=5"
```

## Local Development
```bash
git clone https://github.com/unnatii14/fish-classifier-backend.git
cd fish-classifier-backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Tech Stack
- **Framework**: FastAPI with Uvicorn
- **ML Model**: EfficientNet-B0 (PyTorch)
- **Similarity**: Cosine similarity with scikit-learn
- **Deployment**: Railway (CPU-optimized)
- **Python**: 3.10+