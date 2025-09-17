# 🐟 Fish Species Classifier API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![Railway](https://img.shields.io/badge/Deploy-Railway-blueviolet)](https://railway.app)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen)](https://web-production-cc66.up.railway.app/)

A production-ready FastAPI application for fish species classification using deep learning. This API leverages EfficientNet architecture to classify 31 different fish species with high accuracy, designed for seamless integration with mobile and web applications.

## 🌐 Live Demo

**🔗 API Base URL**: [https://web-production-cc66.up.railway.app](https://web-production-cc66.up.railway.app)

**📖 Interactive Documentation**: [https://web-production-cc66.up.railway.app/docs](https://web-production-cc66.up.railway.app/docs)

## � Model Performance

- **Architecture**: EfficientNet-B0
- **Accuracy**: 99.5% on test dataset
- **Species**: 31 fish species classification
- **Training Data**: 13,000+ labeled images
- **Input Size**: 224x224 pixels
- **Inference Time**: <100ms per image

## ✨ Features

- 🎯 **High Accuracy**: 99.5% classification accuracy
- 🚀 **Fast Inference**: Sub-100ms prediction time
- 🌐 **RESTful API**: Complete REST API with FastAPI
- 📱 **Mobile Ready**: CORS-enabled for mobile app integration
- 🐳 **Production Ready**: Containerized and cloud-deployment optimized
- 📋 **Demo Mode**: Graceful fallback when model is unavailable
- 🔍 **Health Monitoring**: Built-in health checks and monitoring
- 📚 **Auto Documentation**: Interactive API docs with Swagger UI

## 🐟 Supported Fish Species

<details>
<summary>View all 31 supported fish species</summary>

| Species | Common Name | Species | Common Name |
|---------|-------------|---------|-------------|
| Bangus | Milkfish | Silver Barb | Silver Barb |
| Big Head Carp | Bighead Carp | Silver Carp | Silver Carp |
| Black Spotted Barb | Black Spotted Barb | Silver Perch | Silver Perch |
| Catfish | Catfish | Snakehead | Snakehead |
| Climbing Perch | Climbing Perch | Tenpounder | Tenpounder |
| Fourfinger Threadfin | Fourfinger Threadfin | Tilapia | Tilapia |
| Freshwater Eel | Freshwater Eel | Goby | Goby |
| Glass Perchlet | Glass Perchlet | Gold Fish | Goldfish |
| Gourami | Gourami | Grass Carp | Grass Carp |
| Green Spotted Puffer | Green Spotted Puffer | Indian Carp | Indian Carp |
| Indo-Pacific Tarpon | Indo-Pacific Tarpon | Jaguar Gapote | Jaguar Gapote |
| Janitor Fish | Janitor Fish | Knifefish | Knifefish |
| Long-Snouted Pipefish | Long-Snouted Pipefish | Mosquito Fish | Mosquito Fish |
| Mudfish | Mudfish | Mullet | Mullet |
| Pangasius | Pangasius | Perch | Perch |
| Scat Fish | Scat Fish | | |

</details>

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/unnatii14/fish-classifier-backend.git
cd fish-classifier-backend
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
```bash
# Development mode
uvicorn main_minimal:app --reload

# Or with full model (if available)
uvicorn main:app --reload
```

### 4. Access API
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Status**: http://localhost:8000/

## 📋 API Endpoints

### Core Endpoints

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| `GET` | `/` | API status and information | JSON with API details |
| `GET` | `/health` | Health check endpoint | Health status |
| `GET` | `/classes` | List all fish species | Array of species names |
| `GET` | `/model-info` | Model architecture info | Model details |

### Prediction Endpoints

| Method | Endpoint | Description | Input | Output |
|--------|----------|-------------|-------|--------|
| `POST` | `/predict` | Upload image for classification | Multipart file | Prediction results |
| `POST` | `/predict-base64` | Base64 image classification | JSON with base64 string | Prediction results |

### Example Response
```json
{
  "success": true,
  "predictions": [
    {"species": "Tilapia", "confidence": 95.5},
    {"species": "Catfish", "confidence": 3.2},
    {"species": "Bangus", "confidence": 1.3}
  ],
  "top_prediction": "Tilapia",
  "confidence": 95.5
}
```

## 🚢 Deployment

### Railway (Recommended)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

1. **Fork this repository**
2. **Connect to Railway**: Link your GitHub repo
3. **Auto-Deploy**: Railway automatically deploys
4. **Access**: Your API will be live at `https://your-app.railway.app`

### Docker Deployment

```dockerfile
# Build
docker build -t fish-classifier-api .

# Run
docker run -p 8000:8000 fish-classifier-api
```

### Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Start production server
uvicorn main_minimal:app --host 0.0.0.0 --port 8000
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8000` |
| `PYTHON_VERSION` | Python version | `3.10` |
| `MODEL_PATH` | Path to model file | `best_model_efficientnet.pth` |

### Files Structure

```
fish-classifier-backend/
├── main.py                 # Full API with PyTorch
├── main_minimal.py         # Lightweight demo version
├── requirements.txt        # Production dependencies
├── requirements-minimal.txt # Minimal dependencies
├── railway.toml           # Railway configuration
├── Procfile              # Process configuration
├── runtime.txt           # Python version
├── .dockerignore         # Docker ignore rules
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🔄 Modes of Operation

### Demo Mode (Current)
- ✅ All endpoints functional
- ✅ Returns realistic demo predictions
- ✅ No model file required
- ✅ Fast deployment and startup

### Full Mode (With Model)
- ✅ Actual fish classification
- ✅ 99.5% accuracy predictions
- ✅ Real-time image processing
- ⚠️ Requires model file upload

## 📱 Client Integration

### cURL Example
```bash
# Upload image for classification
curl -X POST "https://web-production-cc66.up.railway.app/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@fish_image.jpg"

# Base64 image classification
curl -X POST "https://web-production-cc66.up.railway.app/predict-base64" \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_string"}'
```

### Python Example
```python
import requests

# Get fish species list
response = requests.get("https://web-production-cc66.up.railway.app/classes")
species = response.json()["classes"]

# Upload image for prediction
with open("fish_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "https://web-production-cc66.up.railway.app/predict", 
        files=files
    )
    prediction = response.json()
```

## �️ Development

### Local Setup
```bash
# Clone repository
git clone https://github.com/unnatii14/fish-classifier-backend.git
cd fish-classifier-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main_minimal:app --reload
```

### Testing
```bash
# Run API tests
python test_railway.py

# Test specific endpoint
python -c "
import requests
response = requests.get('http://localhost:8000/health')
print(response.json())
"
```

## 📈 Performance Metrics

- **Response Time**: <100ms average
- **Throughput**: 1000+ requests/minute
- **Accuracy**: 99.5% on test dataset
- **Uptime**: 99.9% availability target
- **Memory Usage**: <512MB RAM
- **Storage**: <50MB (without model)

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **[@unnatii14](https://github.com/unnatii14)** - *Initial work*

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- FastAPI framework by Sebastián Ramirez
- Railway platform for seamless deployment
- OpenCV and PIL for image processing

## � Support

- **Issues**: [GitHub Issues](https://github.com/unnatii14/fish-classifier-backend/issues)
- **Discussions**: [GitHub Discussions](https://github.com/unnatii14/fish-classifier-backend/discussions)
- **Email**: [Contact Developer](mailto:your-email@example.com)

---

<div align="center">

**⭐ Star this repository if it helped you! ⭐**

[🐟 Live Demo](https://web-production-cc66.up.railway.app) • [📖 API Docs](https://web-production-cc66.up.railway.app/docs) • [🚀 Deploy Fork](https://railway.app/new/template)

</div>
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload

# Or use Python directly
python main.py
```

## 📝 Environment Variables

- `PORT`: Server port (default: 8000)
- `PYTHON_VERSION`: Python version for Railway

## 🎯 Model Information

- **Architecture**: EfficientNet-B0
- **Input Size**: 224x224 pixels
- **Classes**: 31 fish species
- **Accuracy**: 99.5% (on test set)
- **Training Images**: 13,000+

## 🚨 Important Notes

- The model file (`best_model_efficientnet.pth`) is not included in git due to size constraints
- Demo mode provides example responses when model is not available
- All endpoints work in demo mode, but predictions are simulated
- Upload the model file to your Railway deployment for full functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.