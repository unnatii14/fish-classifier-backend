# test_api.py - Test script to verify the fixed API works correctly
import requests
import base64
import json
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

def create_test_image():
    """Create a simple test image for testing"""
    # Create a simple RGB image
    img = Image.new('RGB', (224, 224), color='blue')
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img, img_byte_arr

def test_api_local():
    """Test the API locally"""
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"Health status: {response.status_code}")
        print(f"Health response: {response.json()}")
        
        # Test classes endpoint
        print("\nTesting classes endpoint...")
        response = requests.get(f"{base_url}/classes")
        print(f"Classes status: {response.status_code}")
        classes_data = response.json()
        print(f"Number of classes: {classes_data['total']}")
        print(f"First 5 classes: {classes_data['classes'][:5]}")
        
        # Test prediction with base64
        print("\nTesting base64 prediction...")
        test_img, test_img_bytes = create_test_image()
        encoded_img = base64.b64encode(test_img_bytes).decode('utf-8')
        
        payload = {"image": encoded_img}
        response = requests.post(f"{base_url}/predict-base64", json=payload)
        print(f"Prediction status: {response.status_code}")
        if response.status_code == 200:
            pred_data = response.json()
            print(f"Success: {pred_data['success']}")
            if pred_data['success']:
                print(f"Top prediction: {pred_data['top_prediction']}")
                print(f"Confidence: {pred_data['confidence']}%")
                print(f"All predictions: {pred_data['predictions']}")
            else:
                print(f"Demo mode: {pred_data.get('demo_mode', False)}")
        
        print("\n✅ API test completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_api_local()