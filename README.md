# 🐟 Fish Classifier API

A FastAPI-based fish species classifier using EfficientNet deep learning model. This API can classify 31 different fish species.

## 🚀 Railway Deployment

This project is optimized for Railway deployment and includes demo mode functionality.

### Quick Deploy to Railway

1. **Fork this repository**
2. **Connect to Railway**: Go to [Railway](https://railway.app) and connect your GitHub repo
3. **Deploy**: Railway will automatically detect the configuration and deploy
4. **Demo Mode**: The API will run in demo mode without the model file

### Adding the Model File (For Full Functionality)

Since the model file is too large for git, you have several options:

1. **Upload via Railway Dashboard**: Use Railway's file upload feature
2. **Download from URL**: Modify the code to download the model on startup
3. **Use cloud storage**: Store the model in AWS S3, Google Cloud, etc.

## 🔧 Configuration Files

- `railway.toml`: Railway deployment configuration
- `Procfile`: Process file for web server
- `requirements.txt`: Python dependencies (Railway-optimized)
- `runtime.txt`: Python version specification

## 📋 API Endpoints

- `GET /`: API status and information
- `GET /health`: Health check endpoint
- `GET /classes`: List all fish species
- `POST /predict`: Upload image for classification
- `POST /predict-base64`: Send base64 encoded image
- `GET /model-info`: Model information

## 🐟 Supported Fish Species

The model can classify 31 fish species including:
- Bangus, Tilapia, Catfish
- Big Head Carp, Grass Carp, Silver Carp
- Gourami, Perch, Snakehead
- And 22 more species...

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload

# Or use Python directly
python main.py
```

## 📝 Environment Variables

- `PORT`: Server port (default: 8000)
- `PYTHON_VERSION`: Python version for Railway

## 🎯 Model Information

- **Architecture**: EfficientNet-B0
- **Input Size**: 224x224 pixels
- **Classes**: 31 fish species
- **Accuracy**: 99.5% (on test set)
- **Training Images**: 13,000+

## 🚨 Important Notes

- The model file (`best_model_efficientnet.pth`) is not included in git due to size constraints
- Demo mode provides example responses when model is not available
- All endpoints work in demo mode, but predictions are simulated
- Upload the model file to your Railway deployment for full functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.