#!/usr/bin/env python3
"""
Railway Deployment Test Script
Tests the optimized Fish Classifier API for Railway deployment
"""

import requests
import json
import base64
import time
from PIL import Image
import io

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (224, 224), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    return img_b64

def test_railway_deployment(base_url="https://your-app.railway.app"):
    """Test Railway deployment endpoints"""
    print("🚀 Testing Railway Fish Classifier API Deployment")
    print(f"Base URL: {base_url}")
    print("=" * 50)
    
    # Test 1: Root endpoint (health check)
    print("1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint: {data.get('message', 'OK')}")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Model loaded: {data.get('model_loaded', False)}")
            print(f"   Version: {data.get('version', 'unknown')}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 2: Health check
    print("\n2. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: {data.get('status', 'unknown')}")
            print(f"   Railway optimized: {data.get('railway_optimized', False)}")
            if data.get('demo_mode'):
                print("   ⚠️  Running in demo mode")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 3: API Info
    print("\n3. Testing info endpoint...")
    try:
        response = requests.get(f"{base_url}/info", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Info: {data.get('name', 'unknown')}")
            print(f"   Platform: {data.get('deployment_platform', 'unknown')}")
            print(f"   Optimization: {data.get('optimization', 'none')}")
        else:
            print(f"❌ Info endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Info endpoint error: {e}")
    
    # Test 4: Fish Classification
    print("\n4. Testing fish classification...")
    try:
        test_image = create_test_image()
        payload = {"image": test_image}
        
        response = requests.post(
            f"{base_url}/predict-base64", 
            json=payload, 
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Classification: {data.get('success', False)}")
            if data.get('demo_mode'):
                print("   ⚠️  Demo mode response")
            else:
                print(f"   Top prediction: {data.get('top_prediction', 'unknown')}")
                print(f"   Confidence: {data.get('confidence', 0)}%")
        else:
            print(f"❌ Classification failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Classification error: {e}")
    
    # Test 5: Similar Images
    print("\n5. Testing similar images...")
    try:
        test_image = create_test_image()
        payload = {"image": test_image}
        
        response = requests.post(
            f"{base_url}/find-similar-base64", 
            json=payload, 
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Similar images: {data.get('success', False)}")
            if data.get('demo_mode'):
                print("   ⚠️  Demo mode response")
            else:
                similar_count = len(data.get('similar_images', []))
                print(f"   Found {similar_count} similar images")
        else:
            print(f"❌ Similar images failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Similar images error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Railway deployment test completed!")

def test_local():
    """Test local deployment"""
    print("🏠 Testing Local Deployment")
    test_railway_deployment("http://localhost:8000")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "local":
            test_local()
        else:
            test_railway_deployment(sys.argv[1])
    else:
        print("Usage:")
        print("  python test_railway_deployment.py local                    # Test local")
        print("  python test_railway_deployment.py https://your-app.railway.app  # Test Railway")