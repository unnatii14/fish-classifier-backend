"""
Test API with Server Check
This script waits for the server to be ready before testing
"""

import requests
import time
import json
import base64
from PIL import Image
import io

def wait_for_server(url="http://127.0.0.1:8001", max_wait=30):
    """Wait for the server to be ready"""
    print("🔍 Checking if server is ready...")
    
    for i in range(max_wait):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"   ✅ Server is ready! (took {i+1} seconds)")
                return True
        except:
            pass
        
        print(f"   ⏳ Waiting for server... ({i+1}/{max_wait})")
        time.sleep(1)
    
    print(f"   ❌ Server not ready after {max_wait} seconds")
    return False

def create_test_image():
    """Create a test image for API testing"""
    img = Image.new('RGB', (224, 224), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def test_complete_api():
    """Test all API functionality"""
    API_URL = "http://127.0.0.1:8001"
    
    print("\n🧪 Testing Fish Classifier API...")
    print("=" * 50)
    
    # Wait for server
    if not wait_for_server(API_URL):
        print("\n❌ Server is not running!")
        print("\n💡 To start the server:")
        print("   1. Run: python start_api_server.py")
        print("   2. Or manually: python -m uvicorn main:app --host 127.0.0.1 --port 8001")
        return False
    
    test_image = create_test_image()
    
    # Test Classification
    print("\n🐟 Testing Fish Classification...")
    try:
        response = requests.post(
            f"{API_URL}/predict-base64",
            json={"image": test_image, "top_k": 3},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            print(f"   ✅ Classification working! Got {len(predictions)} predictions:")
            
            for i, pred in enumerate(predictions[:3], 1):
                species = pred.get('species', 'Unknown')
                confidence = pred.get('confidence', 0)
                print(f"      {i}. {species}: {confidence:.1%}")
        else:
            print(f"   ❌ Classification failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Classification error: {e}")
        return False
    
    # Test Similar Images
    print("\n🔍 Testing Similar Images...")
    try:
        response = requests.post(
            f"{API_URL}/find-similar-base64",
            json={"image": test_image, "top_k": 5},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            similar_images = data.get('similar_images', [])
            print(f"   ✅ Similar images working! Found {len(similar_images)} similar images:")
            
            for i, img in enumerate(similar_images[:3], 1):
                species = img.get('species_name', 'Unknown')
                similarity = img.get('similarity', 0)
                print(f"      {i}. {species}: {similarity:.1%}")
            
            # Test fish image loading for the first similar image
            if similar_images:
                first_index = similar_images[0].get('index', 0)
                print(f"\n🖼️  Testing Fish Image Loading (index {first_index})...")
                
                img_response = requests.get(f"{API_URL}/fish-image/{first_index}", timeout=10)
                if img_response.status_code == 200:
                    img_data = img_response.json()
                    if img_data.get('success'):
                        species = img_data.get('species_name', 'Unknown')
                        has_image = 'image_base64' in img_data and len(img_data.get('image_base64', '')) > 50
                        print(f"      ✅ Fish image loaded: {species}")
                        print(f"      🎨 Has image data: {has_image}")
                    else:
                        print("      ⚠️  Fish image endpoint returned success=False")
                else:
                    print(f"      ❌ Fish image failed: {img_response.status_code}")
        else:
            print(f"   ❌ Similar images failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Similar images error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All API Tests Passed!")
    print("\n📱 Your API is ready for Flutter integration!")
    print("\n💡 Next steps:")
    print("   1. Open simple_test_interface.html in your browser")
    print("   2. Upload a real fish image to test")
    print("   3. Verify similar images show fish shapes")
    print("   4. Use these endpoints in your Flutter app:")
    print("      - POST /predict-base64 (classification)")
    print("      - POST /find-similar-base64 (similar images)")
    
    return True

if __name__ == "__main__":
    print("Fish Classifier API - Complete Test")
    print("This will test your API once the server is running")
    print()
    
    success = test_complete_api()
    
    if not success:
        print("\n💡 Troubleshooting:")
        print("   1. Make sure the server is running in another window")
        print("   2. Check for any error messages in the server window")
        print("   3. Try restarting the server")
    
    input("\nPress Enter to exit...")