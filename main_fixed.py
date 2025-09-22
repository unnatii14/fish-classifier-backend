# main_fixed.py - Fish Classifier API Fixed to match Streamlit implementation
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Set environment variables for memory optimization
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

app = FastAPI(title="Fish Classifier API", description="Classify fish species and find similar images", version="2.0.0")

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set torch to use single thread for smaller memory footprint
torch.set_num_threads(1)

# Fish class names (31 species) - Same order as Streamlit
CLASS_NAMES = [
    'Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch',
    'Fourfinger Threadfin', 'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish',
    'Gourami', 'Grass Carp', 'Green Spotted Puffer', 'Indian Carp', 'Indo-Pacific Tarpon',
    'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish', 'Mosquito Fish',
    'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
    'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia'
]

# Create label mapping (id_to_label) matching Streamlit approach
id_to_label = {i: CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}

# Global variables
model = None
feature_extractor = None
transform = None
val_embeddings = None
image_paths = None

def load_model():
    """Load the trained PyTorch model - EXACTLY like Streamlit"""
    global model, feature_extractor
    try:
        model_path = "best_model_efficientnet.pth"
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model file '{model_path}' not found. API will run in demo mode.")
            return False
        
        # Load exactly like Streamlit - Classification model
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        model.eval()
        
        # Feature extractor (remove classifier) - EXACTLY like Streamlit
        feature_extractor = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        feature_extractor.classifier[1] = nn.Linear(feature_extractor.classifier[1].in_features, len(CLASS_NAMES))
        feature_extractor.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        feature_extractor.classifier = nn.Identity()
        feature_extractor.eval()
        
        print("✅ Model and feature extractor loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def load_embeddings():
    """Load precomputed embeddings and image paths"""
    global val_embeddings, image_paths
    try:
        # Load embeddings
        if os.path.exists("val_embeddings.npy"):
            val_embeddings = np.load("val_embeddings.npy")
            print(f"✅ Loaded {val_embeddings.shape[0]} embeddings")
        else:
            print("⚠️  val_embeddings.npy not found")
            return False
            
        # Load image paths - handle both .npy and .txt formats
        if os.path.exists("val_image_paths.npy"):
            image_paths_array = np.load("val_image_paths.npy", allow_pickle=True)
            image_paths = image_paths_array.tolist() if hasattr(image_paths_array, 'tolist') else list(image_paths_array)
            print(f"✅ Loaded {len(image_paths)} image paths from .npy file")
        elif os.path.exists("val_image_paths.txt"):
            with open("val_image_paths.txt", "r") as f:
                image_paths = [line.strip() for line in f.readlines()]
            print(f"✅ Loaded {len(image_paths)} image paths from .txt file")
        else:
            print("⚠️  val_image_paths file not found, creating dummy paths")
            image_paths = [f"image_{i}.jpg" for i in range(len(val_embeddings))]
            
        return True
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False

def setup_transforms():
    """Setup image preprocessing transforms - EXACTLY like Streamlit"""
    global transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_topk_predictions(image, k=5):
    """Get top-k predictions - EXACTLY like Streamlit"""
    if model is None:
        return None
        
    try:
        with torch.no_grad():
            input_tensor = transform(image).unsqueeze(0)
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Get top-k indices
            top_indices = probs.argsort()[::-1][:k]
            
            predictions = []
            for i in top_indices:
                predictions.append({
                    "species": id_to_label[int(i)],
                    "confidence": round(float(probs[i]) * 100, 2)
                })
                
            return predictions
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def get_similar_images(query_img, top_k=5):
    """Find similar images - EXACTLY like Streamlit"""
    if feature_extractor is None or val_embeddings is None:
        return None, None
        
    try:
        # Reshape embeddings if needed
        embeddings = val_embeddings
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
            
        with torch.no_grad():
            input_tensor = transform(query_img).unsqueeze(0)
            query_features = feature_extractor(input_tensor).cpu().numpy()
            
            if query_features.ndim > 2:
                query_features = query_features.reshape(query_features.shape[0], -1)
        
        # Compute similarities
        similarities = cosine_similarity(query_features, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return top_indices, similarities[top_indices]
    except Exception as e:
        print(f"Error finding similar images: {e}")
        return None, None

# Initialize everything on module load
setup_transforms()
model_loaded = load_model()
embeddings_loaded = load_embeddings()

@app.get("/")
async def root():
    return {
        "message": "🐟 Fish Classifier API with Similarity Search (Fixed Version)",
        "status": "running",
        "model_loaded": model is not None,
        "feature_extractor_loaded": feature_extractor is not None,
        "embeddings_loaded": val_embeddings is not None,
        "total_classes": len(CLASS_NAMES),
        "total_embeddings": len(val_embeddings) if val_embeddings is not None else 0,
        "note": "Fixed to match Streamlit implementation exactly",
        "endpoints": {
            "predict": "/predict - Upload image for classification",
            "predict_base64": "/predict-base64 - Send base64 image",
            "find_similar": "/find-similar - Find similar images",
            "classes": "/classes - Get all fish species",
            "health": "/health - API health check"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_extractor_loaded": feature_extractor is not None,
        "embeddings_loaded": val_embeddings is not None
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
        return {
            "success": False,
            "demo_mode": True,
            "message": "Model not loaded - running in demo mode",
            "demo_predictions": [
                {"species": "Tilapia", "confidence": 95.5},
                {"species": "Catfish", "confidence": 3.2},
                {"species": "Bangus", "confidence": 1.3}
            ]
        }
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Get predictions using same method as Streamlit
        predictions = get_topk_predictions(image, k=5)
        
        if predictions is None:
            raise HTTPException(status_code=500, detail="Error processing image")
        
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
        return {
            "success": False,
            "demo_mode": True,
            "message": "Model not loaded - running in demo mode",
            "demo_predictions": [
                {"species": "Bangus", "confidence": 92.1},
                {"species": "Tilapia", "confidence": 5.8},
                {"species": "Catfish", "confidence": 2.1}
            ]
        }
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Get predictions using same method as Streamlit
        predictions = get_topk_predictions(image, k=5)
        
        if predictions is None:
            raise HTTPException(status_code=500, detail="Error processing image")
        
        return {
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0]["species"],
            "confidence": predictions[0]["confidence"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/find-similar")
async def find_similar(file: UploadFile = File(...), top_k: int = 5):
    """Find top-k most similar images to the uploaded image"""
    if feature_extractor is None or val_embeddings is None:
        return {
            "success": False,
            "demo_mode": True,
            "message": "Feature extractor or embeddings not loaded - running in demo mode",
            "similar_images": [
                {"image_path": "images/bangus_1.jpg", "similarity_score": 0.99},
                {"image_path": "images/bangus_2.jpg", "similarity_score": 0.97},
                {"image_path": "images/bangus_3.jpg", "similarity_score": 0.95},
                {"image_path": "images/bangus_4.jpg", "similarity_score": 0.93},
                {"image_path": "images/bangus_5.jpg", "similarity_score": 0.91}
            ]
        }
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Find similar images using same method as Streamlit
        top_indices, scores = get_similar_images(image, top_k=top_k)
        
        if top_indices is None:
            raise HTTPException(status_code=500, detail="Failed to find similar images")
        
        similar_images = []
        for idx, sim in zip(top_indices, scores):
            similar_images.append({
                "image_path": image_paths[idx],
                "similarity_score": round(float(sim), 4),
                "index": int(idx)
            })
        
        return {
            "success": True,
            "query_processed": True,
            "top_k": top_k,
            "similar_images": similar_images
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar images: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model": "EfficientNet-B0",
        "version": "Fixed to match Streamlit",
        "classes": len(CLASS_NAMES),
        "input_size": "224x224",
        "model_loaded": model is not None,
        "feature_extractor_loaded": feature_extractor is not None,
        "embeddings_loaded": val_embeddings is not None,
        "total_embeddings": len(val_embeddings) if val_embeddings is not None else 0,
        "species_list": CLASS_NAMES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)