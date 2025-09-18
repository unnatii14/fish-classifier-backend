# Fish Classifier API

A FastAPI application for fish species classification and similarity search. Classifies 31 different fish species using deep learning and finds similar images.

## Live Demo
**API URL**: https://web-production-cc66.up.railway.app  
**Documentation**: https://web-production-cc66.up.railway.app/docs

## Features
- 31 fish species classification
- Image similarity search
- RESTful API with FastAPI
- Real-time image processing
- CORS enabled for web/mobile apps

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/health` | Health check |
| GET | `/classes` | List all fish species |
| POST | `/predict` | Upload image for prediction |
| POST | `/predict-base64` | Base64 image prediction |
| POST | `/find-similar` | Find similar images |

## Quick Test
```bash
# Get fish species
curl https://web-production-cc66.up.railway.app/classes

# Find similar images (upload a fish image)
curl -X POST "https://web-production-cc66.up.railway.app/find-similar" \
  -F "file=@fish_image.jpg" \
  -F "top_k=5"
```

## Local Setup
```bash
git clone https://github.com/unnatii14/fish-classifier-backend.git
cd fish-classifier-backend
pip install -r requirements.txt
uvicorn main_minimal:app --reload
```

## Deploy to Railway
1. Fork this repository
2. Connect to Railway
3. Deploy automatically

## Tech Stack
- FastAPI
- Python 3.10
- Railway deployment