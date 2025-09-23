# main_railway_optimized.py - Railway-optimized Fish Classifier API
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import io
import base64
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Railway optimization environment variables
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

app = FastAPI(
    title="Fish Classifier API", 
    description="Railway-optimized fish species classifier", 
    version="2.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set torch optimizations for Railway
torch.set_num_threads(1)
torch.backends.cudnn.enabled = False

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
val_embeddings = None
image_paths = None
model_loaded = False
embeddings_loaded = False

def load_model():
    """Load the fish classification model with Railway optimization"""
    global model, model_loaded
    try:
        model_path = "best_model_efficientnet.pth"
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file {model_path} not found - running in demo mode")
            model_loaded = False
            return None
        
        # Log model file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Loading model - size: {file_size:.1f} MB")
        
        # Load model with Railway optimization
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Linear(model.classifier[1].in_features, 31)
        
        # Load with CPU mapping for Railway
        model_state = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(model_state, strict=False)
        model.eval()
        
        logger.info("✅ Model loaded successfully!")
        model_loaded = True
        return model
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model_loaded = False
        return None

def load_embeddings():
    """Load precomputed embeddings and image paths"""
    global val_embeddings, image_paths, embeddings_loaded
    try:
        # Load embeddings
        if os.path.exists("val_embeddings.npy"):
            val_embeddings = np.load("val_embeddings.npy")
            logger.info(f"✅ Loaded {val_embeddings.shape[0]} embeddings")
        else:
            logger.warning("⚠️ val_embeddings.npy not found")
            embeddings_loaded = False
            return False
            
        # Load image paths
        if os.path.exists("val_image_paths.npy"):
            image_paths_array = np.load("val_image_paths.npy", allow_pickle=True)
            image_paths = image_paths_array.tolist() if hasattr(image_paths_array, 'tolist') else list(image_paths_array)
            logger.info(f"✅ Loaded {len(image_paths)} image paths")
        else:
            logger.warning("⚠️ val_image_paths file not found, creating dummy paths")
            image_paths = [f"image_{i}.jpg" for i in range(len(val_embeddings))]
            
        embeddings_loaded = True
        return True
    except Exception as e:
        logger.error(f"❌ Error loading embeddings: {e}")
        embeddings_loaded = False
        return False

def setup_transforms():
    """Setup image preprocessing transforms"""
    global transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_features(image):
    """Extract features from an image using the model"""
    if model is None:
        return None
    
    try:
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            # Create feature extractor
            feature_model = models.efficientnet_b0(weights=None)
            feature_model.classifier[1] = nn.Linear(feature_model.classifier[1].in_features, 31)
            feature_model.load_state_dict(model.state_dict(), strict=False)
            feature_model.classifier = nn.Identity()
            feature_model.eval()
            
            features = feature_model(input_tensor)
            
        return features.numpy()
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

# Initialize everything
logger.info("🚀 Starting Railway initialization...")
setup_transforms()

# Load model and embeddings
try:
    load_model()
    load_embeddings()
    logger.info("✅ Railway initialization completed")
except Exception as e:
    logger.error(f"❌ Railway initialization error: {e}")

@app.get("/")
async def root():
    """Root endpoint - Railway health check"""
    return {
        "message": "🐟 Fish Classifier API - Railway Deployment",
        "status": "healthy",
        "model_loaded": model_loaded,
        "embeddings_loaded": embeddings_loaded,
        "total_species": len(CLASS_NAMES),
        "deployment": "Railway Optimized",
        "version": "2.1.0",
        "health_check": True
    }

@app.get("/health")
async def health_check():
    """Dedicated health check endpoint for Railway monitoring"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": "2025-09-23",
            "railway_optimized": True,
            "model_ready": model_loaded,
            "embeddings_ready": embeddings_loaded,
            "species_count": len(CLASS_NAMES)
        }
        
        # If critical components are missing, still return 200 but indicate demo mode
        if not model_loaded:
            health_status["demo_mode"] = True
            health_status["message"] = "Running in demo mode - model not loaded"
        
        return health_status
    except Exception as e:
        # Still return 200 for Railway health check
        return {
            "status": "degraded",
            "error": str(e),
            "demo_mode": True
        }

@app.get("/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": "Fish Classifier API",
        "version": "2.1.0",
        "description": "Railway-optimized fish species classification",
        "supported_species": len(CLASS_NAMES),
        "model_architecture": "EfficientNet-B0",
        "deployment_platform": "Railway",
        "optimization": "Memory optimized for Railway deployment",
        "endpoints": {
            "classification": "/predict-base64",
            "similarity_search": "/find-similar-base64",
            "health": "/health",
            "info": "/info"
        }
    }

@app.get("/classes")
async def get_classes():
    """Get all fish species names"""
    return {
        "classes": CLASS_NAMES,
        "total": len(CLASS_NAMES)
    }

@app.post("/predict-base64")
async def predict_fish_base64(data: Dict, top_k: int = 5):
    """Predict fish species from base64 encoded image"""
    if not model_loaded or model is None:
        # Return demo response for Railway health check compatibility
        return {
            "success": True,
            "demo_mode": True,
            "message": "Model not loaded - running in demo mode",
            "predictions": [
                {"species": "Bangus", "confidence": 92.1},
                {"species": "Tilapia", "confidence": 5.8},
                {"species": "Catfish", "confidence": 2.1}
            ],
            "top_prediction": "Bangus",
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
            
            topk_prob, topk_indices = torch.topk(probabilities, min(top_k, len(CLASS_NAMES)))
            
            predictions = []
            for i in range(min(top_k, len(CLASS_NAMES))):
                predictions.append({
                    "species": CLASS_NAMES[topk_indices[i].item()],
                    "confidence": round(topk_prob[i].item() * 100, 2)
                })
        
        return {
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0]["species"],
            "confidence": predictions[0]["confidence"],
            "top_k": top_k
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/find-similar-base64")
async def find_similar_base64(data: Dict, top_k: int = 5):
    """Find similar images using base64 encoded image"""
    if not model_loaded or not embeddings_loaded or model is None or val_embeddings is None:
        # Return demo response
        return {
            "success": True,
            "demo_mode": True,
            "message": "Model or embeddings not loaded - running in demo mode",
            "similar_images": [
                {"image_path": "images/bangus_1.jpg", "similarity": 0.99, "species_name": "Bangus"},
                {"image_path": "images/bangus_2.jpg", "similarity": 0.97, "species_name": "Bangus"},
                {"image_path": "images/tilapia_1.jpg", "similarity": 0.95, "species_name": "Tilapia"}
            ]
        }
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Extract features
        query_features = extract_features(image)
        if query_features is None:
            raise HTTPException(status_code=500, detail="Failed to extract features")
        
        # Compute similarity
        similarities = cosine_similarity(query_features, val_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_images = []
        for idx in top_indices:
            image_path = image_paths[idx]
            species_name = image_path.split('/')[-2] if '/' in image_path else 'Unknown'
            
            similar_images.append({
                "image_path": image_path,
                "similarity": round(float(similarities[idx]), 4),
                "index": int(idx),
                "species_name": species_name
            })
        
        return {
            "success": True,
            "top_k": top_k,
            "similar_images": similar_images
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar images: {str(e)}")

# Simple startup test endpoint
@app.get("/startup-test")
async def startup_test():
    """Test endpoint for Railway startup verification"""
    return {
        "status": "API is running",
        "railway_deployment": True,
        "timestamp": "2025-09-23"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Railway deployment on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")