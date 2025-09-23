"""
Railway-optimized version of the Fish Classifier API
This version has better error handling and Railway-specific optimizations
"""

import os
import sys
import traceback
from contextlib import asynccontextmanager

# Import FastAPI and core dependencies first
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Early logging setup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for models
model = None
transform = None
CLASS_NAMES = []
val_embeddings = None
val_image_paths = None

def safe_import_and_setup():
    """Safely import all dependencies and set up the application"""
    try:
        # Import FastAPI and core dependencies
        from fastapi import FastAPI, HTTPException, UploadFile, File
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, StreamingResponse
        from pydantic import BaseModel
        
        # Import ML dependencies
        import torch
        import torchvision.transforms as transforms
        from torchvision import models
        from torchvision.models import EfficientNet_B0_Weights
        
        # Import other dependencies
        from PIL import Image
        import numpy as np
        import base64
        import io
        from sklearn.metrics.pairwise import cosine_similarity
        
        logger.info("✅ All dependencies imported successfully")
        return True, None
        
    except Exception as e:
        logger.error(f"❌ Failed to import dependencies: {e}")
        return False, str(e)

def initialize_models():
    """Initialize all models and data with better error handling"""
    global model, transform, CLASS_NAMES, val_embeddings, val_image_paths
    
    try:
        # Set up transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info("✅ Transforms initialized")
        
        # Load model
        model_path = "best_model_efficientnet.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
            
        # Import torch here to avoid early import issues
        import torch
        from torchvision import models
        from torchvision.models import EfficientNet_B0_Weights
        
        # Railway optimization: limit threads
        torch.set_num_threads(1)
        
        # Create model architecture
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 31)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        logger.info("✅ Model loaded successfully")
        
        # Load class names
        CLASS_NAMES = [
            'Anglerfish', 'Barracuda', 'Eel', 'Flounder', 'Gar', 'Goby', 'Grouper',
            'Grunt', 'Hammerhead', 'Jack', 'Lionfish', 'Parrotfish', 'Pufferfish',
            'Ray', 'Remora', 'Shark', 'Snapper', 'Sole', 'Surgeonfish', 'Tang',
            'Tarpon', 'Tuna', 'Wrasse', 'Yellowtail', 'Angelfish', 'Bass',
            'Butterfly Fish', 'Clownfish', 'Damselfish', 'Filefish', 'Triggerfish'
        ]
        
        # Load embeddings
        try:
            val_embeddings = np.load("val_embeddings.npy")
            val_image_paths = np.load("val_image_paths.npy")
            logger.info(f"✅ Loaded {len(val_embeddings)} embeddings")
        except Exception as e:
            logger.warning(f"⚠️ Could not load embeddings: {e}")
            val_embeddings = None
            val_image_paths = None
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        logger.error(traceback.format_exc())
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Fish Classifier API...")
    
    # Import dependencies
    success, error = safe_import_and_setup()
    if not success:
        logger.error(f"Failed to set up application: {error}")
        # Don't exit, let the app start with limited functionality
    
    # Initialize models
    model_success = initialize_models()
    if model_success:
        logger.info("✅ All models initialized successfully")
    else:
        logger.warning("⚠️ Some models failed to initialize")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Fish Classifier API...")

# Create FastAPI app
app = FastAPI(
    title="Fish Classifier API",
    description="Railway-optimized Fish Species Classification API using EfficientNet-B0",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ImageRequest(BaseModel):
    image: str
    top_k: int = 3

class SimilarRequest(BaseModel):
    image: str
    top_k: int = 5

# Health check endpoints
@app.get("/")
async def root():
    """Root endpoint with deployment status"""
    return {
        "message": "Fish Classifier API - Railway Optimized",
        "status": "running",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "embeddings_loaded": val_embeddings is not None,
        "deployment": "Railway optimized v2"
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "classes_available": len(CLASS_NAMES) if CLASS_NAMES else 0,
        "embeddings_loaded": val_embeddings is not None,
        "embeddings_count": len(val_embeddings) if val_embeddings is not None else 0
    }

@app.get("/info")
async def api_info():
    """API information and documentation"""
    return {
        "api_name": "Fish Classifier API",
        "version": "2.0.0",
        "description": "Railway-optimized Fish species classification using EfficientNet-B0",
        "model_classes": len(CLASS_NAMES) if CLASS_NAMES else 0,
        "similarity_embeddings": len(val_embeddings) if val_embeddings is not None else 0,
        "endpoints": {
            "predict": "/predict-base64",
            "similar": "/find-similar-base64",
            "health": "/health",
            "docs": "/docs"
        },
        "deployment": "Railway optimized v2",
        "class_names": CLASS_NAMES[:5] if CLASS_NAMES else []  # Show first 5 classes
    }

def process_base64_image(base64_image: str):
    """Process base64 image with error handling"""
    try:
        # Remove data URL prefix if present
        if base64_image.startswith('data:image'):
            base64_image = base64_image.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.post("/predict-base64")
async def predict_fish_base64(request: ImageRequest):
    """Predict fish species from base64 image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process image
        image = process_base64_image(request.image)
        
        # Import torch here to avoid import issues
        import torch
        
        # Transform and predict
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get top k predictions
        top_prob, top_class = torch.topk(probabilities, request.top_k)
        
        predictions = []
        for i in range(request.top_k):
            predictions.append({
                "species": CLASS_NAMES[top_class[i].item()],
                "confidence": round(top_prob[i].item() * 100, 2)
            })
        
        return {
            "success": True,
            "predictions": predictions,
            "model_version": "EfficientNet-B0",
            "total_classes": len(CLASS_NAMES)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/find-similar-base64")
async def find_similar_base64(request: SimilarRequest):
    """Find similar fish images using base64 input"""
    if val_embeddings is None:
        return {
            "success": False,
            "message": "Similarity search not available - embeddings not loaded",
            "similar_images": []
        }
    
    try:
        # For now, return a placeholder response
        # In a full implementation, you'd extract features and find similar images
        return {
            "success": True,
            "similar_images": [
                {
                    "image_path": f"sample_fish_{i}.jpg",
                    "similarity_score": round(0.9 - i * 0.1, 2),
                    "species": CLASS_NAMES[i % len(CLASS_NAMES)]
                }
                for i in range(min(request.top_k, 5))
            ],
            "message": "Similarity search using placeholder data"
        }
        
    except Exception as e:
        logger.error(f"Similarity search error: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

# Railway deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ImportError:
        logger.error("Uvicorn not available for local development")