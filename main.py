# main.py - Complete Fish Classifier API
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import base64
from typing import Dict
import json

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

# Global variables
model = None
transform = None

def load_model():
    """Load the trained PyTorch model"""
    global model
    try:
        # Create model architecture
        model = models.efficientnet_b0(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 31)
        
        # Check if model file exists
        import os
        model_path = "best_model_efficientnet.pth"
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model file '{model_path}' not found. API will run in demo mode.")
            print("💡 For full functionality, upload your model file to the deployment.")
            return False
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 API will continue running in demo mode.")
        return False

def setup_transforms():
    """Setup image preprocessing transforms"""
    global transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Initialize model and transforms on startup
@app.on_event("startup")
async def startup_event():
    setup_transforms()
    if not load_model():
        print("⚠️  Running in DEMO MODE - Model not loaded")
        print("📝 All endpoints work except actual prediction")
        print("🔧 To enable predictions: upload 'best_model_efficientnet.pth' to your deployment")

@app.get("/")
async def root():
    return {
        "message": "🐟 Fish Classifier API",
        "status": "running",
        "model_loaded": model is not None,
        "mode": "Production" if model is not None else "Demo Mode - Model file missing",
        "total_classes": len(CLASS_NAMES),
        "note": "Upload 'best_model_efficientnet.pth' to enable predictions" if model is None else "All features available",
        "endpoints": {
            "predict": "/predict - Upload image for classification" + (" (DEMO MODE)" if model is None else ""),
            "predict_base64": "/predict-base64 - Send base64 image" + (" (DEMO MODE)" if model is None else ""),
            "classes": "/classes - Get all fish species",
            "health": "/health - API health check"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/classes")
async def get_classes():
    """Get all fish species names"""
    return {
        "classes": CLASS_NAMES,
        "total": len(CLASS_NAMES)
    }

@app.post("/predict")
async def predict_fish(file: UploadFile = File(...)):
    """Predict fish species from uploaded image"""
    if model is None:
        # Return demo response instead of error
        return {
            "success": False,
            "demo_mode": True,
            "message": "API running in demo mode - model file not found",
            "note": "Upload 'best_model_efficientnet.pth' to enable actual predictions",
            "demo_predictions": [
                {"species": "Tilapia", "confidence": 95.5},
                {"species": "Catfish", "confidence": 3.2},
                {"species": "Bangus", "confidence": 1.3}
            ],
            "top_prediction": "Tilapia (Demo)",
            "confidence": 95.5
        }
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, 3)
            
            predictions = []
            for i in range(3):
                predictions.append({
                    "species": CLASS_NAMES[top3_indices[i].item()],
                    "confidence": round(top3_prob[i].item() * 100, 2)
                })
        
        return {
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0]["species"],
            "confidence": predictions[0]["confidence"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict-base64")
async def predict_fish_base64(data: Dict):
    """Predict fish species from base64 encoded image"""
    if model is None:
        # Return demo response instead of error
        return {
            "success": False,
            "demo_mode": True,
            "message": "API running in demo mode - model file not found",
            "note": "Upload 'best_model_efficientnet.pth' to enable actual predictions",
            "demo_predictions": [
                {"species": "Bangus", "confidence": 92.1},
                {"species": "Tilapia", "confidence": 5.8},
                {"species": "Catfish", "confidence": 2.1}
            ],
            "top_prediction": "Bangus (Demo)",
            "confidence": 92.1
        }
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, 3)
            
            predictions = []
            for i in range(3):
                predictions.append({
                    "species": CLASS_NAMES[top3_indices[i].item()],
                    "confidence": round(top3_prob[i].item() * 100, 2)
                })
        
        return {
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0]["species"],
            "confidence": predictions[0]["confidence"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model": "EfficientNet-B0",
        "classes": len(CLASS_NAMES),
        "input_size": "224x224",
        "accuracy": "99.5%",
        "total_images_trained": "13,000",
        "species_list": CLASS_NAMES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)