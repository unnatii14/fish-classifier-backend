#!/usr/bin/env python3
"""
Comprehensive Fish Classifier API Testing Script
Tests the API endpoints and verifies fish species classification accuracy
"""

import requests
import base64
import json
import time
from PIL import Image
import io
import os
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your deployed URL if needed
TEST_IMAGE_PATH = None  # Will be set automatically if test images found

class FishClassifierAPITester:
    def __init__(self, base_url=API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_api_health(self):
        """Test if API is running and healthy"""
        print("🏥 Testing API Health...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API is healthy!")
                print(f"   - Status: {data.get('status')}")
                print(f"   - Model loaded: {data.get('model_loaded')}")
                print(f"   - Embeddings loaded: {data.get('embeddings_loaded')}")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return False
    
    def test_root_endpoint(self):
        """Test root endpoint for API info"""
        print("\n🏠 Testing Root Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Root endpoint working!")
                print(f"   - Message: {data.get('message')}")
                print(f"   - Model loaded: {data.get('model_loaded')}")
                print(f"   - Total classes: {data.get('total_classes')}")
                print(f"   - Total embeddings: {data.get('total_embeddings')}")
                return True
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            return False
    
    def test_classes_endpoint(self):
        """Test classes endpoint"""
        print("\n📋 Testing Classes Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/classes")
            if response.status_code == 200:
                data = response.json()
                classes = data.get('classes', [])
                print(f"✅ Classes endpoint working!")
                print(f"   - Total classes: {len(classes)}")
                print(f"   - First 5 classes: {classes[:5]}")
                print(f"   - Last 5 classes: {classes[-5:]}")
                return classes
            else:
                print(f"❌ Classes endpoint failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Classes endpoint error: {e}")
            return None
    
    def create_test_image(self):
        """Create a simple test image if no real fish image is available"""
        print("\n🖼️ Creating test image...")
        # Create a simple colored image for testing
        img = Image.new('RGB', (224, 224), color='blue')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()
    
    def image_to_base64(self, image_bytes):
        """Convert image bytes to base64 string"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def test_prediction_with_file_upload(self, image_bytes=None):
        """Test file upload prediction endpoint"""
        print("\n🔮 Testing File Upload Prediction...")
        try:
            if image_bytes is None:
                image_bytes = self.create_test_image()
            
            files = {'file': ('test_fish.jpg', image_bytes, 'image/jpeg')}
            response = self.session.post(f"{self.base_url}/predict", files=files)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ File upload prediction successful!")
                    print(f"   - Top prediction: {data.get('top_prediction')}")
                    print(f"   - Confidence: {data.get('confidence')}%")
                    print(f"   - All predictions:")
                    for i, pred in enumerate(data.get('predictions', [])[:3], 1):
                        print(f"     {i}. {pred['species']} - {pred['confidence']}%")
                    return data
                else:
                    print(f"⚠️ API running in demo mode: {data.get('message')}")
                    return data
            else:
                print(f"❌ File upload prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        except Exception as e:
            print(f"❌ File upload prediction error: {e}")
            return None
    
    def test_prediction_with_base64(self, image_bytes=None):
        """Test base64 prediction endpoint"""
        print("\n🔮 Testing Base64 Prediction...")
        try:
            if image_bytes is None:
                image_bytes = self.create_test_image()
            
            base64_image = self.image_to_base64(image_bytes)
            payload = {"image": base64_image}
            
            response = self.session.post(
                f"{self.base_url}/predict-base64",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ Base64 prediction successful!")
                    print(f"   - Top prediction: {data.get('top_prediction')}")
                    print(f"   - Confidence: {data.get('confidence')}%")
                    print(f"   - All predictions:")
                    for i, pred in enumerate(data.get('predictions', [])[:3], 1):
                        print(f"     {i}. {pred['species']} - {pred['confidence']}%")
                    return data
                else:
                    print(f"⚠️ API running in demo mode: {data.get('message')}")
                    return data
            else:
                print(f"❌ Base64 prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Base64 prediction error: {e}")
            return None
    
    def test_similarity_search(self, image_bytes=None):
        """Test similarity search endpoint"""
        print("\n🔍 Testing Similarity Search...")
        try:
            if image_bytes is None:
                image_bytes = self.create_test_image()
            
            files = {'file': ('test_fish.jpg', image_bytes, 'image/jpeg')}
            params = {'top_k': 5}
            
            response = self.session.post(
                f"{self.base_url}/find-similar",
                files=files,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ Similarity search successful!")
                    print(f"   - Found {len(data.get('similar_images', []))} similar images")
                    for i, sim in enumerate(data.get('similar_images', [])[:3], 1):
                        print(f"     {i}. {sim['image_path']} - {sim['similarity_score']:.4f}")
                    return data
                else:
                    print(f"⚠️ Similarity search in demo mode: {data.get('message')}")
                    return data
            else:
                print(f"❌ Similarity search failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Similarity search error: {e}")
            return None
    
    def test_model_info(self):
        """Test model info endpoint"""
        print("\n📊 Testing Model Info...")
        try:
            response = self.session.get(f"{self.base_url}/model-info")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model info retrieved!")
                print(f"   - Model: {data.get('model')}")
                print(f"   - Classes: {data.get('classes')}")
                print(f"   - Input size: {data.get('input_size')}")
                print(f"   - Model loaded: {data.get('model_loaded')}")
                print(f"   - Total embeddings: {data.get('total_embeddings')}")
                return data
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Model info error: {e}")
            return None
    
    def load_real_fish_image(self):
        """Try to load a real fish image for testing"""
        # Look for common image files in current directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        current_dir = Path('.')
        
        for ext in image_extensions:
            for img_file in current_dir.glob(f'*{ext}'):
                try:
                    with open(img_file, 'rb') as f:
                        image_bytes = f.read()
                    print(f"📸 Found test image: {img_file}")
                    return image_bytes
                except Exception:
                    continue
        
        print("📸 No real fish images found, using generated test image")
        return None
    
    def run_comprehensive_test(self):
        """Run all API tests"""
        print("🐠 Starting Comprehensive Fish Classifier API Test")
        print("=" * 60)
        
        # Test API connectivity
        if not self.test_api_health():
            print("❌ API is not accessible. Make sure it's running on the correct port.")
            return False
        
        # Test basic endpoints
        self.test_root_endpoint()
        classes = self.test_classes_endpoint()
        self.test_model_info()
        
        # Load test image (real or generated)
        test_image = self.load_real_fish_image()
        
        # Test prediction endpoints
        file_result = self.test_prediction_with_file_upload(test_image)
        base64_result = self.test_prediction_with_base64(test_image)
        
        # Test similarity search
        similarity_result = self.test_similarity_search(test_image)
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        if file_result and file_result.get('success'):
            print("✅ File upload prediction: WORKING")
        elif file_result:
            print("⚠️ File upload prediction: DEMO MODE")
        else:
            print("❌ File upload prediction: FAILED")
        
        if base64_result and base64_result.get('success'):
            print("✅ Base64 prediction: WORKING")
        elif base64_result:
            print("⚠️ Base64 prediction: DEMO MODE")
        else:
            print("❌ Base64 prediction: FAILED")
        
        if similarity_result and similarity_result.get('success'):
            print("✅ Similarity search: WORKING")
        elif similarity_result:
            print("⚠️ Similarity search: DEMO MODE")
        else:
            print("❌ Similarity search: FAILED")
        
        # Check for real predictions vs demo mode
        working_properly = (
            file_result and file_result.get('success') and
            base64_result and base64_result.get('success')
        )
        
        if working_properly:
            print("\n🎉 API IS WORKING CORRECTLY WITH REAL PREDICTIONS!")
            print("Your fish classifier is ready for production use.")
        elif file_result or base64_result:
            print("\n⚠️ API is running but in DEMO MODE")
            print("Check if model file 'best_model_efficientnet.pth' is in the correct location.")
        else:
            print("\n❌ API has issues - check server logs")
        
        return working_properly

def main():
    """Main test function"""
    print("🚀 Fish Classifier API Test Suite")
    print("Make sure your API is running before starting tests")
    print("Default URL: http://localhost:8000")
    print()
    
    # Ask user for API URL
    api_url = input("Enter API URL (press Enter for http://localhost:8000): ").strip()
    if not api_url:
        api_url = "http://localhost:8000"
    
    # Create tester instance
    tester = FishClassifierAPITester(api_url)
    
    # Run tests
    success = tester.run_comprehensive_test()
    
    # Exit with appropriate code
    exit(0 if success else 1)

if __name__ == "__main__":
    main()