# Fish Classifier API

A production-ready FastAPI application for fish species classification and similarity search using deep learning. Deployed on Railway with EfficientNet-B0 architecture for accurate fish identification across 31 species.

## Live Demo
**API URL**: https://web-production-cc66.up.railway.app  
**Interactive Docs**: https://web-production-cc66.up.railway.app/docs

## Features
- **31 Fish Species Classification** - Accurate identification using EfficientNet-B0
- **Image Similarity Search** - Find visually similar fish using deep embeddings
- **RESTful API** - Well-documented FastAPI with automatic OpenAPI docs
- **Production Ready** - Optimized for Railway deployment with health checks
- **CORS Enabled** - Ready for web and mobile app integration

## API Endpoints

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| GET | `/` | API status and info | - |
| GET | `/health` | Health check | - |
| GET | `/classes` | List all 31 fish species | - |
| POST | `/predict` | Classify fish image | `file`: image |
| POST | `/predict-base64` | Classify base64 image | `image`: base64 string |
| POST | `/find-similar` | Find similar fish images | `file`: image, `top_k`: int |

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