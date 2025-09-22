#!/usr/bin/env python3
"""
Direct test of the similar images functionality
"""

import requests
import base64
from PIL import Image
import io
import json

def test_similar_images_detailed():
    print('🧪 Detailed Similar Images Test')
    print('=' * 50)
    
    # Create test image (simulating a fish image)
    img = Image.new('RGB', (224, 224), color=(100, 150, 200))  # Blue-ish like fish
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    
    print(f'🖼️  Created test image (base64 length: {len(base64_image)} characters)')
    
    # Test the API
    api_url = 'http://127.0.0.1:8002'
    
    try:
        print(f'🌐 Testing API: {api_url}/find-similar-base64')
        response = requests.post(
            f'{api_url}/find-similar-base64?top_k=5',
            json={'image': base64_image},
            timeout=20
        )
        
        print(f'📡 Response Status: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            print(f'✅ Success: {data.get("success")}')
            
            similar_images = data.get('similar_images', [])
            print(f'🔍 Similar images found: {len(similar_images)}')
            
            if similar_images:
                print('📋 Similar images details:')
                for i, img_data in enumerate(similar_images, 1):
                    filename = img_data['image_path'].split('/')[-1]
                    similarity = img_data['similarity']
                    print(f'   {i}. {filename}')
                    print(f'      Similarity: {similarity:.4f}')
                    print(f'      Index: {img_data["index"]}')
                    print(f'      Percentage: {abs(similarity * 100):.1f}%')
                    print()
                
                # Test what the web interface receives
                print('🌐 Web Interface Data Format:')
                print(json.dumps(data, indent=2))
            else:
                print('❌ No similar images returned')
        else:
            print(f'❌ API Error: {response.status_code}')
            print(f'Response: {response.text}')
            
    except Exception as e:
        print(f'❌ Connection Error: {e}')

if __name__ == "__main__":
    test_similar_images_detailed()