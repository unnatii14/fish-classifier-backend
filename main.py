# main.py - Fish Classifier API with Similarity Search
import os
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

def load_model():
    """Load the trained PyTorch model - FIXED to match Streamlit"""
    global model
    try:
        # Create model architecture with pretrained ImageNet weights (same as Streamlit)
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 31)
        
        # Check if model file exists
        model_path = "best_model_efficientnet.pth"
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model file '{model_path}' not found. API will run in demo mode.")
            return False
        
        # Load weights with strict=False (same as Streamlit)
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        model.eval()
        
        print("✅ Model loaded successfully with fixed approach!")
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
            # Create a feature extractor model (same approach as Streamlit)
            feature_model = models.efficientnet_b0(weights=None)
            feature_model.classifier[1] = nn.Linear(feature_model.classifier[1].in_features, 31)
            feature_model.load_state_dict(model.state_dict(), strict=False)
            feature_model.classifier = nn.Identity()  # Remove classifier for features
            feature_model.eval()
            
            # Extract features
            features = feature_model(input_tensor)
            
        return features.numpy()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Initialize everything on module load
setup_transforms()
model_loaded = load_model()
embeddings_loaded = load_embeddings()

@app.get("/")
async def root():
    return {
        "message": "🐟 Fish Classifier API with Similarity Search",
        "status": "running",
        "model_loaded": model is not None,
        "embeddings_loaded": val_embeddings is not None,
        "total_classes": len(CLASS_NAMES),
        "total_embeddings": len(val_embeddings) if val_embeddings is not None else 0,
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
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 5 predictions to match Streamlit
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            predictions = []
            for i in range(5):
                predictions.append({
                    "species": CLASS_NAMES[top5_indices[i].item()],
                    "confidence": round(top5_prob[i].item() * 100, 2)
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
async def predict_fish_base64(data: Dict, top_k: int = 5):
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
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top_k predictions (configurable)
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

@app.post("/find-similar")
async def find_similar(file: UploadFile = File(...), top_k: int = 5):
    """Find top-k most similar images to the uploaded image"""
    if model is None or val_embeddings is None:
        return {
            "success": False,
            "demo_mode": True,
            "message": "Model or embeddings not loaded - running in demo mode",
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
        
        # Extract features from uploaded image
        query_features = extract_features(image)
        if query_features is None:
            raise HTTPException(status_code=500, detail="Failed to extract features from image")
        
        # Compute cosine similarity with all embeddings
        similarities = cosine_similarity(query_features, val_embeddings)[0]
        
        # Get top-k most similar images
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_images = []
        for idx in top_indices:
            similar_images.append({
                "image_path": image_paths[idx],
                "similarity_score": round(float(similarities[idx]), 4),
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
        "classes": len(CLASS_NAMES),
        "input_size": "224x224",
        "model_loaded": model is not None,
        "embeddings_loaded": val_embeddings is not None,
        "total_embeddings": len(val_embeddings) if val_embeddings is not None else 0,
        "species_list": CLASS_NAMES
    }

@app.post("/find-similar-base64")
async def find_similar_base64(data: Dict, top_k: int = 5):
    """Find similar images using base64 encoded image"""
    if model is None or val_embeddings is None or image_paths is None:
        return {
            "success": False,
            "demo_mode": True,
            "message": "Model or embeddings not loaded - running in demo mode",
            "similar_images": [
                {"image_path": "images/bangus_1.jpg", "similarity_score": 0.99},
                {"image_path": "images/bangus_2.jpg", "similarity_score": 0.97},
                {"image_path": "images/bangus_3.jpg", "similarity_score": 0.95},
                {"image_path": "images/bangus_4.jpg", "similarity_score": 0.93},
                {"image_path": "images/bangus_5.jpg", "similarity_score": 0.91}
            ]
        }
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Extract features from uploaded image
        query_features = extract_features(image)
        if query_features is None:
            raise HTTPException(status_code=500, detail="Failed to extract features from image")
        
        # Compute cosine similarity with all embeddings
        similarities = cosine_similarity(query_features, val_embeddings)[0]
        
        # Get top-k most similar images
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_images = []
        for idx in top_indices:
            image_path = image_paths[idx]
            species_name = image_path.split('/')[-2] if '/' in image_path else 'Unknown'
            filename = image_path.split('/')[-1] if '/' in image_path else image_path
            
            similar_images.append({
                "image_path": image_path,
                "similarity": round(float(similarities[idx]), 4),
                "index": int(idx),
                "species_name": species_name,
                "filename": filename,
                "image_url": f"/fish-image/{idx}"
            })
        
        return {
            "success": True,
            "query_processed": True,
            "top_k": top_k,
            "similar_images": similar_images
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar images: {str(e)}")

@app.get("/fish-image/{image_index}")
async def get_fish_image(image_index: int):
    """Get a fish image for similar images display - creates species-specific placeholder since actual images are not available locally"""
    if image_paths is None or image_index >= len(image_paths):
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        # Extract species info from the image path
        image_path = image_paths[image_index]
        species_name = image_path.split('/')[-2] if '/' in image_path else 'Unknown'
        filename = image_path.split('/')[-1] if '/' in image_path else image_path
        
        # Create species-specific fish representations
        from PIL import Image, ImageDraw, ImageFont
        import io
        import base64
        import hashlib
        
        # Generate species-specific colors based on species name
        species_hash = hashlib.md5(species_name.encode()).hexdigest()
        
        # Define fish-specific color schemes
        fish_colors = {
            'Bangus': [(70, 130, 180), (100, 149, 237), (135, 206, 250)],  # Blue tones
            'Salmon': [(255, 160, 122), (255, 127, 80), (255, 99, 71)],     # Salmon/orange tones
            'Trout': [(46, 139, 87), (60, 179, 113), (144, 238, 144)],     # Green tones
            'Tuna': [(25, 25, 112), (72, 61, 139), (106, 90, 205)],        # Dark blue tones
            'Cod': [(192, 192, 192), (169, 169, 169), (211, 211, 211)],    # Silver/gray tones
            'Bass': [(139, 69, 19), (160, 82, 45), (205, 133, 63)],        # Brown tones
        }
        
        # Select color scheme based on species
        colors = fish_colors.get(species_name, [(70, 130, 180), (100, 149, 237), (135, 206, 250)])
        
        # Create a 120x80 fish-like image
        img = Image.new('RGB', (120, 80), color=colors[0])
        draw = ImageDraw.Draw(img)
        
        # Draw fish body (ellipse)
        draw.ellipse([10, 20, 90, 60], fill=colors[1])
        draw.ellipse([15, 25, 85, 55], fill=colors[2])
        
        # Draw fish tail
        draw.polygon([(85, 30), (110, 20), (110, 60), (85, 50)], fill=colors[1])
        
        # Draw fish eye
        draw.ellipse([20, 30, 30, 40], fill=(255, 255, 255))
        draw.ellipse([23, 33, 27, 37], fill=(0, 0, 0))
        
        # Add species label
        try:
            font_size = 10
            draw.text((5, 5), species_name[:8], fill=(255, 255, 255), font=None)
        except:
            pass
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "image_base64": f"data:image/png;base64,{img_base64}",
            "species_name": species_name,
            "filename": filename,
            "image_path": image_path,
            "note": "This is a generated placeholder. Actual fish images are not available in this deployment."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.get("/image/{image_index}")
async def get_image(image_index: int):
    """Serve an image by its index in the dataset"""
    if image_paths is None or image_index >= len(image_paths):
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        image_path = image_paths[image_index]
        # For now, return a placeholder since we don't have actual image files
        # In a real deployment, you would serve the actual image file
        return {
            "image_path": image_path,
            "image_index": image_index,
            "note": "Image serving would require actual image files in deployment"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)