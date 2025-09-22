# web_interface.py - Simple Flask web interface for testing the Fish Classifier API
from flask import Flask, render_template_string, request, jsonify
import requests
import base64
import os

app = Flask(__name__)

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐠 Fish Classifier - Web Test Interface</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }
        .content {
            padding: 30px;
        }
        .upload-area {
            border: 2px dashed #dee2e6;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9fa;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #4facfe;
            background: #f0f8ff;
        }
        .upload-button {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            margin-top: 15px;
        }
        .upload-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
        }
        .image-preview {
            max-width: 300px;
            max-height: 300px;
            border-radius: 10px;
            margin: 20px auto;
            display: block;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .results {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }
        .prediction-card, .similarity-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #dee2e6;
        }
        .card-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #495057;
            margin-bottom: 20px;
        }
        .prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            margin-bottom: 10px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #4facfe;
        }
        .prediction-item.top {
            border-left-color: #28a745;
            background: #d4edda;
        }
        .confidence {
            background: #4facfe;
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .prediction-item.top .confidence {
            background: #28a745;
        }
        .api-status {
            background: #e8f5e8;
            border: 1px solid #c3e6c3;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            text-align: center;
        }
        .api-status.error {
            background: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4facfe;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .similar-item {
            background: white;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .similarity-score {
            background: #17a2b8;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }
        @media (max-width: 768px) {
            .results {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐠 Fish Species Classifier</h1>
            <p>Upload a fish image to test the AI classification API</p>
        </div>
        
        <div class="content">
            <div id="apiStatus" class="api-status">
                Checking API connection...
            </div>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area">
                    <div>📁</div>
                    <h3>Upload Fish Image</h3>
                    <p>Choose a fish image to classify</p>
                    <input type="file" id="imageFile" name="image" accept="image/*" style="display: none;" onchange="previewImage(this)">
                    <button type="button" class="upload-button" onclick="document.getElementById('imageFile').click()">
                        Choose Image
                    </button>
                </div>
            </form>
            
            <div id="imagePreview" style="text-align: center;"></div>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <div>Analyzing fish species...</div>
            </div>
            
            <div id="results" class="results" style="display: none;">
                <div class="prediction-card">
                    <div class="card-title">🏆 Species Predictions</div>
                    <div id="predictions"></div>
                </div>
                
                <div class="similarity-card">
                    <div class="card-title">🔍 Similar Images</div>
                    <div id="similarImages"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = 'http://127.0.0.1:8000';

        // Check API status on load
        document.addEventListener('DOMContentLoaded', checkApiStatus);

        async function checkApiStatus() {
            const statusDiv = document.getElementById('apiStatus');
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();
                
                if (data.status === 'healthy') {
                    statusDiv.innerHTML = `✅ API Connected! Model: ${data.model_loaded ? 'Loaded' : 'Not loaded'}`;
                    statusDiv.classList.remove('error');
                } else {
                    throw new Error('API not healthy');
                }
            } catch (error) {
                statusDiv.innerHTML = `❌ API Connection Failed: ${error.message}`;
                statusDiv.classList.add('error');
            }
        }

        function previewImage(input) {
            const preview = document.getElementById('imagePreview');
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.innerHTML = `<img src="${e.target.result}" class="image-preview" alt="Preview">`;
                    classifyImage(input.files[0]);
                };
                reader.readAsDataURL(input.files[0]);
            }
        }

        async function classifyImage(file) {
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            
            loading.style.display = 'block';
            results.style.display = 'none';

            try {
                // Convert to base64
                const base64 = await fileToBase64(file);
                const imageData = base64.split(',')[1];

                // Call prediction API
                const predictionResponse = await fetch(`${API_BASE}/predict-base64`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({image: imageData})
                });

                const predictionData = await predictionResponse.json();

                if (!predictionData.success) {
                    throw new Error(predictionData.message || 'Prediction failed');
                }

                // Call similarity API
                const formData = new FormData();
                formData.append('file', file);
                
                const similarityResponse = await fetch(`${API_BASE}/find-similar?top_k=5`, {
                    method: 'POST',
                    body: formData
                });

                let similarityData = {similar_images: []};
                if (similarityResponse.ok) {
                    similarityData = await similarityResponse.json();
                }

                displayResults(predictionData, similarityData);

            } catch (error) {
                alert(`Error: ${error.message}`);
            } finally {
                loading.style.display = 'none';
            }
        }

        function fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => resolve(reader.result);
                reader.onerror = error => reject(error);
            });
        }

        function displayResults(predictionData, similarityData) {
            const predictions = document.getElementById('predictions');
            const similarImages = document.getElementById('similarImages');
            const results = document.getElementById('results');

            // Display predictions
            predictions.innerHTML = '';
            predictionData.predictions.forEach((pred, index) => {
                predictions.innerHTML += `
                    <div class="prediction-item ${index === 0 ? 'top' : ''}">
                        <span>${pred.species}</span>
                        <span class="confidence">${pred.confidence}%</span>
                    </div>
                `;
            });

            // Display similar images
            similarImages.innerHTML = '';
            if (similarityData.similar_images && similarityData.similar_images.length > 0) {
                similarityData.similar_images.forEach(sim => {
                    const fileName = sim.image_path.split('/').pop() || sim.image_path;
                    similarImages.innerHTML += `
                        <div class="similar-item">
                            <span>🐠</span>
                            <span style="flex-grow: 1; font-size: 0.9em;">${fileName}</span>
                            <span class="similarity-score">${(sim.similarity_score * 100).toFixed(1)}%</span>
                        </div>
                    `;
                });
            } else {
                similarImages.innerHTML = '<p style="color: #6c757d;">No similar images found</p>';
            }

            results.style.display = 'grid';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/test-api')
def test_api():
    """Test endpoint to verify API connectivity"""
    try:
        response = requests.get('http://127.0.0.1:8000/health')
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Fish Classifier Web Interface...")
    print("📍 Open your browser and go to: http://127.0.0.1:5000")
    print("🔗 Make sure your Fish Classifier API is running on port 8000")
    print("=" * 60)
    app.run(debug=True, host='127.0.0.1', port=5000)