#!/usr/bin/env python3
"""
Quick test script to verify the app works
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import app
    print("✅ FastAPI app imported successfully")
    
    # Test basic endpoints exist
    routes = [route.path for route in app.routes]
    expected_routes = ["/", "/health", "/classes", "/predict", "/predict-base64"]
    
    for route in expected_routes:
        if route in routes:
            print(f"✅ Route {route} exists")
        else:
            print(f"❌ Route {route} missing")
    
    print("✅ All tests passed! App should work on Railway.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)