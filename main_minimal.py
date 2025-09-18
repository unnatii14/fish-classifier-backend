# main_minimal.py - Minimal version for initial Railway deployment
import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI(title="Fish Classifier API", description="Classify fish species using EfficientNet", version="1.0.0")

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fish class names (31 species)
CLASS_NAMES = [
    'Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch',
    'Fourfinger Threadfin', 'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish',
    'Gourami', 'Grass Carp', 'Green Spotted Puffer', 'Indian Carp', 'Indo-Pacific Tarpon',
    'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish', 'Mosquito Fish',
    'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
    'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia'
]

@app.get("/")
async def root():
    return {
        "message": "🐟 Fish Classifier API",
        "status": "running",
        "mode": "Minimal Demo Mode",
        "model_loaded": False,
        "total_classes": len(CLASS_NAMES),
        "note": "Running in minimal mode for Railway testing",
        "endpoints": {
            "predict": "/predict - Demo mode only",
            "predict_base64": "/predict-base64 - Demo mode only",
            "classes": "/classes - Get all fish species",
            "health": "/health - API health check"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mode": "minimal_demo",
        "model_loaded": False
    }

@app.get("/classes")
async def get_classes():
    """Get all fish species names"""
    return {
        "classes": CLASS_NAMES,
        "total": len(CLASS_NAMES)
    }

@app.post("/predict")
async def predict_fish():
    """Demo prediction endpoint"""
    return {
        "success": False,
        "demo_mode": True,
        "message": "API running in minimal demo mode",
        "note": "This is a simplified version for Railway testing",
        "demo_predictions": [
            {"species": "Tilapia", "confidence": 95.5},
            {"species": "Catfish", "confidence": 3.2},
            {"species": "Bangus", "confidence": 1.3}
        ],
        "top_prediction": "Tilapia (Demo)",
        "confidence": 95.5
    }

@app.post("/predict-base64")
async def predict_fish_base64():
    """Demo base64 prediction endpoint"""
    return {
        "success": False,
        "demo_mode": True,
        "message": "API running in minimal demo mode",
        "note": "This is a simplified version for Railway testing",
        "demo_predictions": [
            {"species": "Bangus", "confidence": 92.1},
            {"species": "Tilapia", "confidence": 5.8},
            {"species": "Catfish", "confidence": 2.1}
        ],
        "top_prediction": "Bangus (Demo)",
        "confidence": 92.1
    }

@app.post("/find-similar")
async def find_similar(file: UploadFile = File(...), top_k: int = 5):
    """
    Find top-k most similar images to the uploaded image.
    This is a placeholder for demo mode. For real similarity search, use your full model version.
    """
    # --- DEMO RESPONSE ---
    # In your full version, you would:
    # 1. Preprocess the uploaded image
    # 2. Extract embedding using your model
    # 3. Load val_embeddings.npy and val_image_paths.txt
    # 4. Compute cosine similarity
    # 5. Return top_k most similar images
    return {
        "success": True,
        "message": "Demo mode: Similarity search is not available in minimal version.",
        "top_k": top_k,
        "similar_images": [
            {"image_path": "images/bangus_1.jpg", "score": 0.99},
            {"image_path": "images/bangus_2.jpg", "score": 0.97},
            {"image_path": "images/bangus_3.jpg", "score": 0.95},
            {"image_path": "images/bangus_4.jpg", "score": 0.93},
            {"image_path": "images/bangus_5.jpg", "score": 0.91}
        ]
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model": "EfficientNet-B0",
        "mode": "minimal_demo",
        "classes": len(CLASS_NAMES),
        "input_size": "224x224",
        "note": "Model not loaded in minimal mode",
        "species_list": CLASS_NAMES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)