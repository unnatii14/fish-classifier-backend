"""
Server Starter Script
Keeps the API server running in the background
"""

import subprocess
import time
import sys
import os

def start_server():
    """Start the API server in a new console window"""
    print("🚀 Starting Fish Classifier API Server...")
    
    # Kill any existing Python processes
    try:
        subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                      capture_output=True, shell=True)
        print("   Killed existing Python processes")
        time.sleep(2)
    except:
        pass
    
    # Get the current directory
    current_dir = os.getcwd()
    
    # Start the server in a new console window
    cmd = [
        'cmd', '/c', 'start', 'cmd', '/k',
        f'cd /d "{current_dir}" && python -m uvicorn main:app --host 127.0.0.1 --port 8001'
    ]
    
    try:
        subprocess.Popen(cmd, shell=True)
        print("   ✅ Server started in new window!")
        print("   🌐 API should be running at http://127.0.0.1:8001")
        
        # Wait for server to start
        print("   ⏳ Waiting for server to initialize...")
        time.sleep(8)
        
        return True
    except Exception as e:
        print(f"   ❌ Failed to start server: {e}")
        return False

if __name__ == "__main__":
    print("Fish Classifier API - Server Starter")
    print("=" * 40)
    
    if start_server():
        print("\n✅ Server should now be running!")
        print("\nNext steps:")
        print("1. Check the new console window that opened")
        print("2. Run: python quick_api_test.py")
        print("3. Or open: simple_test_interface.html")
        print("\nTo stop the server, close the console window or press Ctrl+C in it.")
    else:
        print("\n❌ Failed to start server. Try running manually:")
        print("python -m uvicorn main:app --host 127.0.0.1 --port 8001")
    
    input("\nPress Enter to exit...")