# GitHub Setup Commands for Fish Classifier Project

Follow these commands step by step to add your project to GitHub:

## Prerequisites
1. Make sure you have Git installed: `git --version`
2. Make sure you have a GitHub account
3. Install GitHub CLI (optional): `gh --version`

## Step 1: Initialize Git Repository
```bash
cd "c:\Users\Hetvi\Downloads\fish-classifier-backend-main (3)\fish-classifier-backend-main"

# Initialize git repository
git init

# Check status
git status
```

## Step 2: Configure Git (if not done before)
```bash
# Set your name and email (replace with your GitHub info)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Check configuration
git config --list
```

## Step 3: Add Files to Git
```bash
# Add all files to staging
git add .

# Check what will be committed
git status

# If you want to exclude large model files, run:
# git rm --cached best_model_efficientnet.pth
# git rm --cached val_embeddings.npy
# git rm --cached val_image_paths.npy
```

## Step 4: Create Initial Commit
```bash
# Create your first commit
git commit -m "Initial commit: Fish Classifier API with EfficientNet-B0

- Add FastAPI backend for fish species classification
- Include 31 species classification model
- Add similarity search functionality  
- Include web testing interface
- Add comprehensive API testing scripts
- CORS enabled for Flutter integration"
```

## Step 5: Create GitHub Repository

### Option A: Using GitHub Web Interface (Recommended)
1. Go to https://github.com
2. Click "New repository" (green button)
3. Repository name: `fish-classifier-backend`
4. Description: `FastAPI fish species classification API with EfficientNet-B0 and similarity search`
5. Choose Public or Private
6. DON'T initialize with README (you already have one)
7. Click "Create repository"

### Option B: Using GitHub CLI (if installed)
```bash
# Create repository using GitHub CLI
gh repo create fish-classifier-backend --public --description "FastAPI fish species classification API with EfficientNet-B0 and similarity search"
```

## Step 6: Connect Local Repository to GitHub
```bash
# Add GitHub repository as remote origin (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/fish-classifier-backend.git

# Verify remote was added
git remote -v
```

## Step 7: Push to GitHub
```bash
# Push your code to GitHub
git branch -M main
git push -u origin main
```

## Step 8: Verify Upload
1. Go to your GitHub repository page
2. Refresh the page
3. You should see all your files uploaded

## Additional Git Commands for Future Updates

### Adding New Changes
```bash
# See what changed
git status

# Add specific files
git add filename.py

# Add all changes
git add .

# Commit changes
git commit -m "Add new feature: description of changes"

# Push to GitHub
git push
```

### Creating Branches for Features
```bash
# Create and switch to new branch
git checkout -b feature/new-feature-name

# Work on your feature, then commit
git add .
git commit -m "Add new feature"

# Push branch to GitHub
git push -u origin feature/new-feature-name

# Switch back to main
git checkout main
```

### Useful Git Commands
```bash
# Check repository status
git status

# See commit history
git log --oneline

# See differences
git diff

# Pull latest changes from GitHub
git pull

# Clone repository (for future use)
git clone https://github.com/YOUR_USERNAME/fish-classifier-backend.git
```

## Repository Structure After Upload
```
fish-classifier-backend/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── best_model_efficientnet.pth (large file)
├── val_embeddings.npy (large file)  
├── val_image_paths.npy
├── simple_test_interface.html
├── test_with_server_check.py
├── start_api_server.py
├── test_everything.bat
└── Other test files...
```

## Important Notes

1. **Large Files**: GitHub has a 100MB file limit. If your model files are too large:
   ```bash
   # Remove large files from git
   git rm --cached best_model_efficientnet.pth
   git commit -m "Remove large model file"
   
   # Use Git LFS for large files (if needed)
   git lfs track "*.pth"
   git add .gitattributes
   git add best_model_efficientnet.pth
   git commit -m "Add model with Git LFS"
   ```

2. **Repository Visibility**: 
   - Public: Anyone can see your code
   - Private: Only you and collaborators can see it

3. **Security**: Never commit API keys, passwords, or sensitive data

## Example Complete Setup
```bash
# Navigate to your project
cd "c:\Users\Hetvi\Downloads\fish-classifier-backend-main (3)\fish-classifier-backend-main"

# Initialize and add files
git init
git add .
git commit -m "Initial commit: Fish Classifier API"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/fish-classifier-backend.git

# Push to GitHub
git branch -M main
git push -u origin main
```

Your project will be live at: `https://github.com/YOUR_USERNAME/fish-classifier-backend`