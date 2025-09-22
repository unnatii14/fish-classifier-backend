#!/usr/bin/env python3
"""
Quick Fish Classifier API Test
Simple script to test your API with a real fish image
"""

import requests
import base64
import json
from PIL import Image
import io

def test_fish_api(api_url="http://localhost:8000", image_path=None):
    """Test the fish classifier API with a real image"""
    
    print(f"🐠 Testing Fish Classifier API at: {api_url}")
    print("=" * 50)
    
    # 1. Check if API is running
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API is running!")
            print(f"   Model loaded: {health_data.get('model_loaded')}")
            print(f"   Status: {health_data.get('status')}")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure your API is running with: uvicorn main:app --reload")
        return False
    
    # 2. Get available classes
    try:
        response = requests.get(f"{api_url}/classes")
        classes = response.json().get('classes', [])
        print(f"\n📋 Available fish species: {len(classes)}")
        print(f"   Examples: {', '.join(classes[:5])}...")
    except Exception as e:
        print(f"⚠️ Could not get classes: {e}")
    
    # 3. Test with image
    if image_path:
        print(f"\n📸 Testing with image: {image_path}")
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Test file upload
            files = {'file': ('fish.jpg', image_bytes, 'image/jpeg')}
            response = requests.post(f"{api_url}/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("✅ Prediction successful!")
                    print(f"   🏆 Top prediction: {result.get('top_prediction')}")
                    print(f"   📊 Confidence: {result.get('confidence')}%")
                    print("\n   Top 5 predictions:")
                    for i, pred in enumerate(result.get('predictions', [])[:5], 1):
                        print(f"   {i}. {pred['species']} - {pred['confidence']}%")
                    
                    # Test base64 as well
                    print("\n🔄 Testing Base64 method...")
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    payload = {"image": base64_image}
                    response2 = requests.post(f"{api_url}/predict-base64", json=payload)
                    
                    if response2.status_code == 200:
                        result2 = response2.json()
                        if result2.get('success'):
                            print(f"✅ Base64 prediction: {result2.get('top_prediction')} ({result2.get('confidence')}%)")
                            
                            # Compare results
                            if result.get('top_prediction') == result2.get('top_prediction'):
                                print("✅ Both methods give consistent results!")
                            else:
                                print("⚠️ Methods give different results - check implementation")
                        else:
                            print("⚠️ Base64 method in demo mode")
                    
                    return True
                else:
                    print("⚠️ API running in demo mode")
                    print(f"   Message: {result.get('message')}")
                    print("   Check if model file is loaded correctly")
                    return False
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except FileNotFoundError:
            print(f"❌ Image file not found: {image_path}")
            return False
        except Exception as e:
            print(f"❌ Error testing with image: {e}")
            return False
    else:
        # Create a test image
        print("\n📸 Creating test image...")
        img = Image.new('RGB', (224, 224), color='lightblue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('test.jpg', img_bytes.getvalue(), 'image/jpeg')}
        response = requests.post(f"{api_url}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Test prediction successful!")
                print(f"   Top prediction: {result.get('top_prediction')}")
                print(f"   Confidence: {result.get('confidence')}%")
                return True
            else:
                print("⚠️ API in demo mode - model not loaded properly")
                return False
        else:
            print(f"❌ Test prediction failed: {response.status_code}")
            return False

if __name__ == "__main__":
    import sys
    
    # Get API URL and image path from command line or use defaults
    api_url = "http://localhost:8000"
    image_path = None
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if len(sys.argv) > 2:
        api_url = sys.argv[2]
    
    print("🚀 Quick Fish API Test")
    print("Usage: python test_api_quick.py [image_path] [api_url]")
    print("Example: python test_api_quick.py bangus.jpg http://localhost:8000")
    print()
    
    success = test_fish_api(api_url, image_path)
    
    if success:
        print("\n🎉 Your Fish Classifier API is working correctly!")
        print("✅ Ready for production use with your Flutter app")
    else:
        print("\n⚠️ Issues detected - check the output above")
        
    exit(0 if success else 1)