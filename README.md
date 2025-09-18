# Fish Classifier API

A simple FastAPI application for fish species classification. Classifies 31 different fish species using deep learning.

## Live Demo
**API URL**: https://web-production-cc66.up.railway.app  
**Documentation**: https://web-production-cc66.up.railway.app/docs

## Features
- 31 fish species classification
- RESTful API with FastAPI
- Demo mode with sample predictions
- CORS enabled for web/mobile apps

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/health` | Health check |
| GET | `/classes` | List all fish species |
| POST | `/predict` | Upload image for prediction |
| POST | `/predict-base64` | Base64 image prediction |

## Quick Test
```bash
curl https://web-production-cc66.up.railway.app/classes
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