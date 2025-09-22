import requests
import json
import base64
from PIL import Image
import io

# Test the fish image endpoint
API_URL = "http://localhost:8002"

def test_fish_image_endpoint():
    """Test the /fish-image/{index} endpoint"""
    print("Testing fish image endpoint...")
    
    # Test with a few different indices
    test_indices = [0, 1, 2, 100, 500]
    
    for index in test_indices:
        try:
            response = requests.get(f"{API_URL}/fish-image/{index}")
            print(f"\nTesting index {index}:")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Success: {data.get('success', False)}")
                print(f"Species: {data.get('species_name', 'Unknown')}")
                print(f"Filename: {data.get('filename', 'Unknown')}")
                print(f"Has image data: {'image_base64' in data and len(data.get('image_base64', '')) > 0}")
                
                # Try to decode and verify the image
                if data.get('image_base64'):
                    try:
                        # Remove data URL prefix if present
                        image_data = data['image_base64']
                        if image_data.startswith('data:image'):
                            image_data = image_data.split(',', 1)[1]
                        
                        # Decode base64
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        print(f"Image size: {image.size}")
                        print(f"Image mode: {image.mode}")
                    except Exception as e:
                        print(f"Error decoding image: {e}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Request failed for index {index}: {e}")

def test_similar_images_with_fish():
    """Test similar images functionality with actual fish classification"""
    print("\n" + "="*50)
    print("Testing similar images with fish classification...")
    
    # Create a test image (placeholder)
    test_image = Image.new('RGB', (224, 224), color='blue')
    
    # Convert to base64
    buffer = io.BytesIO()
    test_image.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    try:
        # Test classification
        response = requests.post(
            f"{API_URL}/predict-base64",
            json={"image": base64_image, "top_k": 3}
        )
        
        print(f"Classification status: {response.status_code}")
        if response.status_code == 200:
            pred_data = response.json()
            print(f"Top prediction: {pred_data.get('predictions', [{}])[0]}")
        
        # Test similar images
        response = requests.post(
            f"{API_URL}/find-similar-base64",
            json={"image": base64_image, "top_k": 5}
        )
        
        print(f"Similar images status: {response.status_code}")
        if response.status_code == 200:
            sim_data = response.json()
            print(f"Found {len(sim_data.get('similar_images', []))} similar images")
            
            for i, similar in enumerate(sim_data.get('similar_images', [])[:3]):
                print(f"  {i+1}. Species: {similar.get('species_name', 'Unknown')}")
                print(f"     Filename: {similar.get('filename', 'Unknown')}")
                print(f"     Similarity: {similar.get('similarity', 0):.3f}")
                print(f"     Index: {similar.get('index', 'Unknown')}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_fish_image_endpoint()
    test_similar_images_with_fish()