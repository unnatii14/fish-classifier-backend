"""
Quick API Test Script
Run this to verify your Fish Classifier API is working correctly
"""

import requests
import json
import base64
from PIL import Image
import io

def create_test_image():
    """Create a simple test image for API testing"""
    # Create a 224x224 blue test image
    img = Image.new('RGB', (224, 224), color='blue')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    return base64_string

def test_api():
    """Test all API endpoints"""
    API_URL = "http://127.0.0.1:8001"
    
    print("🔍 Testing Fish Classifier API...")
    print("=" * 50)
    
    # Test 1: Server Health
    print("\n1. Testing Server Health...")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            print("   ✅ Server is running!")
        else:
            print(f"   ❌ Server error: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot connect to server: {e}")
        print("   💡 Make sure to start the server first!")
        return False
    
    # Test 2: Fish Classification
    print("\n2. Testing Fish Classification...")
    test_image = create_test_image()
    
    try:
        response = requests.post(
            f"{API_URL}/predict-base64",
            json={"image": test_image, "top_k": 3},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            print(f"   ✅ Classification successful!")
            print(f"   📊 Got {len(predictions)} predictions:")
            
            for i, pred in enumerate(predictions, 1):
                species = pred.get('species', 'Unknown')
                confidence = pred.get('confidence', 0)
                print(f"      {i}. {species}: {confidence:.2%}")
        else:
            print(f"   ❌ Classification failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Classification error: {e}")
        return False
    
    # Test 3: Similar Images
    print("\n3. Testing Similar Images...")
    try:
        response = requests.post(
            f"{API_URL}/find-similar-base64",
            json={"image": test_image, "top_k": 5},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            similar_images = data.get('similar_images', [])
            print(f"   ✅ Similar images found!")
            print(f"   📊 Found {len(similar_images)} similar images:")
            
            for i, img in enumerate(similar_images[:3], 1):
                species = img.get('species_name', 'Unknown')
                similarity = img.get('similarity', 0)
                filename = img.get('filename', 'Unknown')
                print(f"      {i}. {species} ({filename}): {similarity:.3f}")
        else:
            print(f"   ❌ Similar images failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Similar images error: {e}")
        return False
    
    # Test 4: Fish Images
    print("\n4. Testing Fish Image Endpoint...")
    try:
        response = requests.get(f"{API_URL}/fish-image/0", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                species = data.get('species_name', 'Unknown')
                has_image = 'image_base64' in data
                print(f"   ✅ Fish image loaded!")
                print(f"   🐟 Species: {species}")
                print(f"   🖼️  Has image data: {has_image}")
            else:
                print("   ⚠️  Fish image endpoint returned success=False")
        else:
            print(f"   ❌ Fish image failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Fish image error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API Test Complete!")
    print("\n💡 Next steps:")
    print("   1. Open simple_test_interface.html in your browser")
    print("   2. Upload a real fish image")
    print("   3. Check if similar images show fish shapes")
    
    return True

if __name__ == "__main__":
    print("Fish Classifier API - Quick Test")
    print("Make sure your API server is running on port 8001")
    print()
    
    success = test_api()
    if success:
        print("\n✅ All tests passed! Your API is working correctly!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")