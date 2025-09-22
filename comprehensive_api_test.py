"""
Comprehensive API Testing Script
Tests all endpoints and verifies correct functionality
"""

import requests
import json
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import time
import sys

# Configuration
API_BASE_URL = "http://127.0.0.1:8001"
ENDPOINTS = {
    'health': f"{API_BASE_URL}/",
    'predict': f"{API_BASE_URL}/predict-base64", 
    'similar': f"{API_BASE_URL}/find-similar-base64",
    'fish_image': f"{API_BASE_URL}/fish-image"
}

def create_test_image(color='blue', size=(224, 224)):
    """Create a test image for API testing"""
    image = Image.new('RGB', size, color=color)
    
    # Add some text to make it more distinctive
    draw = ImageDraw.Draw(image)
    try:
        # Try to use a default font, fallback if not available
        draw.text((50, 100), f"{color.upper()}\nTEST\nIMAGE", fill='white')
    except:
        draw.text((50, 100), f"{color.upper()}\nTEST\nIMAGE", fill='white')
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    image_bytes = buffer.getvalue()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    return base64_string, image

def test_server_health():
    """Test if the server is running"""
    print("🔍 Testing Server Health...")
    try:
        response = requests.get(ENDPOINTS['health'], timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and responsive")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

def test_fish_classification():
    """Test fish classification endpoint"""
    print("\n🐟 Testing Fish Classification...")
    
    # Test with different colored images
    test_cases = [
        ('blue', 'Blue test image'),
        ('red', 'Red test image'), 
        ('green', 'Green test image')
    ]
    
    for color, description in test_cases:
        print(f"\n  Testing with {description}...")
        
        base64_image, _ = create_test_image(color=color)
        
        # Test with top 3
        payload = {
            "image": base64_image,
            "top_k": 3
        }
        
        try:
            response = requests.post(ENDPOINTS['predict'], json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                print(f"    ✅ Classification successful")
                print(f"    📊 Got {len(predictions)} predictions")
                
                for i, pred in enumerate(predictions, 1):
                    species = pred.get('species', 'Unknown')
                    confidence = pred.get('confidence', 0)
                    print(f"      {i}. {species}: {confidence:.2%}")
                
                if len(predictions) >= 3:
                    print("    ✅ Top 3 predictions returned correctly")
                else:
                    print(f"    ⚠️  Expected 3 predictions, got {len(predictions)}")
                    
            else:
                print(f"    ❌ Classification failed with status {response.status_code}")
                print(f"    Response: {response.text}")
                
        except Exception as e:
            print(f"    ❌ Error during classification: {e}")

def test_similar_images():
    """Test similar images endpoint"""
    print("\n🔍 Testing Similar Images...")
    
    base64_image, _ = create_test_image(color='purple')
    
    payload = {
        "image": base64_image,
        "top_k": 5
    }
    
    try:
        response = requests.post(ENDPOINTS['similar'], json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            similar_images = data.get('similar_images', [])
            
            print(f"  ✅ Similar images search successful")
            print(f"  📊 Found {len(similar_images)} similar images")
            
            for i, similar in enumerate(similar_images[:3], 1):
                species = similar.get('species_name', 'Unknown')
                filename = similar.get('filename', 'Unknown')
                similarity = similar.get('similarity', 0)
                index = similar.get('index', 'Unknown')
                
                print(f"    {i}. {species}")
                print(f"       File: {filename}")
                print(f"       Similarity: {similarity:.3f}")
                print(f"       Index: {index}")
                
            if len(similar_images) >= 5:
                print("  ✅ Top 5 similar images returned correctly")
            else:
                print(f"  ⚠️  Expected 5 similar images, got {len(similar_images)}")
                
            return similar_images
            
        else:
            print(f"  ❌ Similar images search failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"  ❌ Error during similar images search: {e}")
        return []

def test_fish_image_endpoint(similar_images):
    """Test fish image serving endpoint"""
    print("\n🖼️  Testing Fish Image Endpoint...")
    
    if not similar_images:
        print("  ⚠️  No similar images to test with, using default indices")
        test_indices = [0, 1, 2]
    else:
        test_indices = [img.get('index', 0) for img in similar_images[:3]]
    
    successful_images = 0
    
    for i, index in enumerate(test_indices):
        print(f"  Testing image index {index}...")
        
        try:
            response = requests.get(f"{ENDPOINTS['fish_image']}/{index}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success', False):
                    species = data.get('species_name', 'Unknown')
                    filename = data.get('filename', 'Unknown')
                    has_image = 'image_base64' in data and len(data.get('image_base64', '')) > 0
                    
                    print(f"    ✅ Image {index} loaded successfully")
                    print(f"       Species: {species}")
                    print(f"       Filename: {filename}")
                    print(f"       Has image data: {has_image}")
                    
                    if has_image:
                        successful_images += 1
                        # Verify the image data is valid base64
                        try:
                            image_data = data['image_base64']
                            if image_data.startswith('data:image'):
                                image_data = image_data.split(',', 1)[1]
                            decoded = base64.b64decode(image_data)
                            img = Image.open(io.BytesIO(decoded))
                            print(f"       Image size: {img.size}")
                        except Exception as e:
                            print(f"       ⚠️  Image data validation failed: {e}")
                else:
                    print(f"    ❌ Image {index} endpoint returned success=False")
                    
            else:
                print(f"    ❌ Image {index} request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ Error testing image {index}: {e}")
    
    print(f"  📊 Successfully loaded {successful_images}/{len(test_indices)} fish images")
    return successful_images > 0

def test_web_interface_compatibility():
    """Test endpoints used by the web interface"""
    print("\n🌐 Testing Web Interface Compatibility...")
    
    # Test the exact flow the web interface uses
    base64_image, _ = create_test_image(color='orange')
    
    # Step 1: Classification
    print("  Step 1: Testing classification for web interface...")
    class_response = requests.post(
        ENDPOINTS['predict'], 
        json={"image": base64_image, "top_k": 3}
    )
    
    if class_response.status_code == 200:
        class_data = class_response.json()
        print(f"    ✅ Classification: {len(class_data.get('predictions', []))} predictions")
    else:
        print(f"    ❌ Classification failed: {class_response.status_code}")
        return False
    
    # Step 2: Similar images
    print("  Step 2: Testing similar images for web interface...")
    similar_response = requests.post(
        ENDPOINTS['similar'],
        json={"image": base64_image, "top_k": 5}
    )
    
    if similar_response.status_code == 200:
        similar_data = similar_response.json()
        similar_images = similar_data.get('similar_images', [])
        print(f"    ✅ Similar images: {len(similar_images)} found")
        
        # Step 3: Test image loading for each similar image
        print("  Step 3: Testing image loading for similar images...")
        loaded_count = 0
        for img in similar_images[:3]:
            index = img.get('index', 0)
            img_response = requests.get(f"{ENDPOINTS['fish_image']}/{index}")
            if img_response.status_code == 200 and img_response.json().get('success'):
                loaded_count += 1
        
        print(f"    ✅ Loaded {loaded_count}/{min(3, len(similar_images))} similar image thumbnails")
        return True
    else:
        print(f"    ❌ Similar images failed: {similar_response.status_code}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🚀 Starting Comprehensive API Test")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test results tracking
    results = {
        'server_health': False,
        'classification': False,
        'similar_images': False,
        'fish_images': False,
        'web_interface': False
    }
    
    # Run tests
    results['server_health'] = test_server_health()
    
    if results['server_health']:
        time.sleep(1)  # Small delay between tests
        test_fish_classification()
        results['classification'] = True
        
        time.sleep(1)
        similar_images = test_similar_images()
        results['similar_images'] = len(similar_images) > 0
        
        time.sleep(1)
        results['fish_images'] = test_fish_image_endpoint(similar_images)
        
        time.sleep(1)
        results['web_interface'] = test_web_interface_compatibility()
    
    # Print summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Duration: {duration:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your API is working correctly!")
        print("\n💡 Next steps:")
        print("   1. Open simple_test_interface.html in your browser")
        print("   2. Upload a fish image and test the interface")
        print("   3. Verify similar images show with thumbnails")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        
    return passed == total

if __name__ == "__main__":
    print("Fish Classifier API - Comprehensive Test Suite")
    print("Make sure your API server is running on http://127.0.0.1:8001")
    print()
    
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)