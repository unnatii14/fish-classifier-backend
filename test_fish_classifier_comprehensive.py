#!/usr/bin/env python3
"""
Comprehensive Fish Classifier API Testing Script
Tests all endpoints and validates correct fish species classification
"""

import requests
import json
import base64
import io
from PIL import Image
import sys
import time

# API Configuration
API_BASE_URL = "http://127.0.0.1:8001"

class FishClassifierTester:
    def __init__(self, base_url=API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        print("🏥 Testing Health Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Health Check: {data}")
                return True
            else:
                print(f"   ❌ Health Check Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Health Check Error: {e}")
            return False
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        print("🏠 Testing Root Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Root Endpoint: {data}")
                return True
            else:
                print(f"   ❌ Root Endpoint Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Root Endpoint Error: {e}")
            return False
    
    def create_test_image(self, color="blue", size=(224, 224)):
        """Create a test image for testing"""
        img = Image.new('RGB', size, color=color)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()
    
    def image_to_base64(self, image_bytes):
        """Convert image bytes to base64 string"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def test_predict_endpoint(self):
        """Test the /predict endpoint with file upload"""
        print("🐟 Testing /predict Endpoint (File Upload)...")
        try:
            # Create test image
            test_image = self.create_test_image()
            
            files = {'file': ('test_fish.jpg', test_image, 'image/jpeg')}
            response = self.session.post(f"{self.base_url}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Predict Endpoint Response:")
                print(f"      Success: {data.get('success', False)}")
                
                if 'predictions' in data and data['predictions']:
                    for i, pred in enumerate(data['predictions'][:3]):  # Show top 3
                        print(f"      {i+1}. {pred.get('species', 'Unknown')}: {pred.get('confidence', 0):.2f}%")
                else:
                    print("      No predictions returned")
                
                return data.get('success', False)
            else:
                print(f"   ❌ Predict Endpoint Failed: {response.status_code}")
                print(f"      Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Predict Endpoint Error: {e}")
            return False
    
    def test_predict_base64_endpoint(self):
        """Test the /predict-base64 endpoint"""
        print("📷 Testing /predict-base64 Endpoint...")
        try:
            # Create test image
            test_image = self.create_test_image(color="green")
            base64_image = self.image_to_base64(test_image)
            
            payload = {"image": base64_image}
            response = self.session.post(
                f"{self.base_url}/predict-base64", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Predict Base64 Endpoint Response:")
                print(f"      Success: {data.get('success', False)}")
                
                if 'predictions' in data and data['predictions']:
                    for i, pred in enumerate(data['predictions'][:3]):  # Show top 3
                        print(f"      {i+1}. {pred.get('species', 'Unknown')}: {pred.get('confidence', 0):.2f}%")
                else:
                    print("      No predictions returned")
                
                return data.get('success', False)
            else:
                print(f"   ❌ Predict Base64 Endpoint Failed: {response.status_code}")
                print(f"      Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Predict Base64 Endpoint Error: {e}")
            return False
    
    def test_find_similar_endpoint(self):
        """Test the /find-similar endpoint"""
        print("🔍 Testing /find-similar Endpoint...")
        try:
            # Create test image
            test_image = self.create_test_image(color="red")
            base64_image = self.image_to_base64(test_image)
            
            payload = {"image": base64_image}
            response = self.session.post(
                f"{self.base_url}/find-similar", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Find Similar Endpoint Response:")
                print(f"      Success: {data.get('success', False)}")
                
                if 'similar_images' in data and data['similar_images']:
                    print(f"      Found {len(data['similar_images'])} similar images")
                    for i, img in enumerate(data['similar_images'][:3]):  # Show top 3
                        print(f"      {i+1}. {img.get('image_path', 'Unknown')}: {img.get('similarity', 0):.4f}")
                else:
                    print("      No similar images found")
                
                return data.get('success', False)
            else:
                print(f"   ❌ Find Similar Endpoint Failed: {response.status_code}")
                print(f"      Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Find Similar Endpoint Error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Comprehensive Fish Classifier API Tests")
        print("=" * 60)
        
        results = {
            'health': self.test_health_endpoint(),
            'root': self.test_root_endpoint(),
            'predict': self.test_predict_endpoint(),
            'predict_base64': self.test_predict_base64_endpoint(),
            'find_similar': self.test_find_similar_endpoint()
        }
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASSED" if passed_test else "❌ FAILED"
            print(f"{test_name.upper():<20} {status}")
            if passed_test:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Your Fish Classifier API is working correctly!")
        elif passed > 0:
            print("⚠️  Some tests passed. API is partially functional.")
        else:
            print("❌ ALL TESTS FAILED. Please check your API server.")
        
        return results

def main():
    """Main function to run tests"""
    print("Fish Classifier API Comprehensive Testing")
    print("This will test all API endpoints to verify correct fish species classification")
    print()
    
    # Check if API server is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"✅ API server is running at {API_BASE_URL}")
    except:
        print(f"❌ API server is not running at {API_BASE_URL}")
        print("Please start your API server first:")
        print("python -m uvicorn main:app --host 127.0.0.1 --port 8001")
        return
    
    # Run tests
    tester = FishClassifierTester()
    results = tester.run_all_tests()
    
    print("\n" + "=" * 60)
    print("📝 NEXT STEPS")
    print("=" * 60)
    print("1. If all tests passed, your API is ready for Flutter integration")
    print("2. Use /predict-base64 endpoint for mobile apps")
    print("3. Test with real fish images using the web interface")
    print("4. Compare results with your Streamlit demo")

if __name__ == "__main__":
    main()