import subprocess
import sys
import os

# Change to the correct directory
os.chdir(r"c:\Users\Hetvi\Downloads\fish-classifier-backend-main (3)\fish-classifier-backend-main")

# Start the server
print("Starting FastAPI server on port 8003...")
try:
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "127.0.0.1", 
        "--port", "8003"
    ], check=True)
except KeyboardInterrupt:
    print("Server stopped by user")
except Exception as e:
    print(f"Error starting server: {e}")