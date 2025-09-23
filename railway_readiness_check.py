#!/usr/bin/env python3
"""
🔍 Railway Readiness Checker for Fish Classifier API
Checks if everything is Railway-friendly and identifies potential issues
"""

import os
import sys
import json
from pathlib import Path

def check_file_sizes():
    """Check file sizes for Railway deployment limits"""
    print("📁 File Size Analysis")
    print("-" * 40)
    
    issues = []
    warnings = []
    
    # Model file check
    model_file = "best_model_efficientnet.pth"
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"✅ Model file: {model_file} ({size_mb:.1f} MB)")
        
        if size_mb > 100:
            issues.append(f"Model file is {size_mb:.1f} MB - Railway prefers < 100MB")
        elif size_mb > 50:
            warnings.append(f"Model file is {size_mb:.1f} MB - consider optimization")
    else:
        warnings.append("Model file not found - will run in demo mode")
    
    # Embedding files check
    for file in ["val_embeddings.npy", "val_image_paths.npy"]:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"✅ Embedding file: {file} ({size_mb:.1f} MB)")
            
            if size_mb > 50:
                warnings.append(f"{file} is {size_mb:.1f} MB - large embedding file")
        else:
            warnings.append(f"Embedding file {file} not found")
    
    return issues, warnings

def check_railway_config():
    """Check Railway configuration files"""
    print("\n⚙️  Railway Configuration")
    print("-" * 40)
    
    issues = []
    warnings = []
    
    # Check railway.toml
    if os.path.exists("railway.toml"):
        print("✅ railway.toml found")
        with open("railway.toml", "r") as f:
            content = f.read()
            if "main_railway_optimized:app" in content:
                print("✅ Uses optimized main file")
            else:
                issues.append("railway.toml should use main_railway_optimized:app")
                
            if "healthcheckPath" in content:
                print("✅ Health check configured")
            else:
                warnings.append("No health check path configured")
    else:
        warnings.append("railway.toml not found - using defaults")
    
    # Check Procfile
    if os.path.exists("Procfile"):
        print("✅ Procfile found")
        with open("Procfile", "r") as f:
            content = f.read()
            if "main_railway_optimized" in content:
                print("✅ Procfile uses optimized version")
            else:
                issues.append("Procfile should use main_railway_optimized")
    else:
        warnings.append("Procfile not found - Railway will use railway.toml")
    
    return issues, warnings

def check_environment_variables():
    """Check for proper environment variable handling"""
    print("\n🌍 Environment Variables")
    print("-" * 40)
    
    issues = []
    warnings = []
    
    # Check main files for PORT usage
    main_files = ["main.py", "main_railway_optimized.py"]
    
    for file in main_files:
        if os.path.exists(file):
            try:
                with open(file, "r", encoding='utf-8') as f:
                    content = f.read()
                    if 'os.environ.get("PORT"' in content:
                        print(f"✅ {file} uses PORT environment variable")
                    else:
                        if file == "main_railway_optimized.py":
                            issues.append(f"{file} missing PORT environment variable")
                        else:
                            warnings.append(f"{file} should use PORT environment variable")
            except UnicodeDecodeError:
                warnings.append(f"Could not read {file} - encoding issue")
    
    return issues, warnings

def check_dependencies():
    """Check requirements.txt for Railway compatibility"""
    print("\n📦 Dependencies")
    print("-" * 40)
    
    issues = []
    warnings = []
    
    if os.path.exists("requirements.txt"):
        print("✅ requirements.txt found")
        
        try:
            with open("requirements.txt", "r", encoding='utf-8') as f:
                deps = f.read()
        except UnicodeDecodeError:
            try:
                with open("requirements.txt", "r", encoding='cp1252') as f:
                    deps = f.read()
            except:
                warnings.append("Could not read requirements.txt - encoding issue")
                return issues, warnings
            
        # Check for CPU-only PyTorch
        if "+cpu" in deps:
            print("✅ Uses CPU-only PyTorch (Railway compatible)")
        else:
            issues.append("Should use CPU-only PyTorch for Railway")
        
        # Check for potential heavy dependencies
        heavy_deps = ["tensorflow", "opencv", "scipy"]
        for dep in heavy_deps:
            if dep in deps.lower():
                warnings.append(f"Heavy dependency detected: {dep}")
        
        # Check for version pinning
        lines = [line.strip() for line in deps.split('\n') if line.strip()]
        pinned = sum(1 for line in lines if '==' in line)
        total = len([line for line in lines if not line.startswith('#') and not line.startswith('-')])
        
        if pinned == total:
            print("✅ All dependencies are version pinned")
        else:
            warnings.append(f"Only {pinned}/{total} dependencies are version pinned")
    else:
        issues.append("requirements.txt not found")
    
    return issues, warnings

def check_api_security():
    """Check for API security considerations"""
    print("\n🔒 API Security")
    print("-" * 40)
    
    issues = []
    warnings = []
    
    # Check for API keys in code
    sensitive_patterns = ["api_key", "API_KEY", "secret", "password", "token"]
    
    for file in ["main.py", "main_railway_optimized.py"]:
        if os.path.exists(file):
            try:
                with open(file, "r", encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                warnings.append(f"Could not read {file} - encoding issue")
                continue
                
                for pattern in sensitive_patterns:
                    if pattern in content and "=" in content:
                        # Simple check - could be improved
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if pattern in line and "=" in line and not line.strip().startswith('#'):
                                warnings.append(f"Potential sensitive data in {file}:{i+1}")
    
    # Check CORS configuration
    main_files = ["main.py", "main_railway_optimized.py"]
    for file in main_files:
        if os.path.exists(file):
            try:
                with open(file, "r", encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                warnings.append(f"Could not read {file} - encoding issue")
                continue
                if 'allow_origins=["*"]' in content:
                    warnings.append(f"{file} allows all CORS origins - consider restricting in production")
                elif "CORSMiddleware" in content:
                    print(f"✅ {file} has CORS configured")
    
    return issues, warnings

def check_railway_optimization():
    """Check Railway-specific optimizations"""
    print("\n🚀 Railway Optimizations")
    print("-" * 40)
    
    issues = []
    warnings = []
    
    if os.path.exists("main_railway_optimized.py"):
        print("✅ Railway-optimized main file exists")
        
        try:
            with open("main_railway_optimized.py", "r", encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            warnings.append("Could not read main_railway_optimized.py - encoding issue")
            return issues, warnings
            
        # Check for memory optimizations
        optimizations = [
            ("OMP_NUM_THREADS", "OpenMP thread limitation"),
            ("MKL_NUM_THREADS", "MKL thread limitation"),
            ("PYTHONUNBUFFERED", "Python output buffering"),
            ("torch.set_num_threads", "PyTorch thread limitation"),
            ("torch.backends.cudnn.enabled = False", "CUDA disabled")
        ]
        
        for opt, desc in optimizations:
            if opt in content:
                print(f"✅ {desc}")
            else:
                warnings.append(f"Missing optimization: {desc}")
        
        # Check for demo mode fallback
        if "demo_mode" in content:
            print("✅ Demo mode fallback implemented")
        else:
            warnings.append("No demo mode fallback for missing models")
            
    else:
        issues.append("main_railway_optimized.py not found")
    
    return issues, warnings

def generate_report():
    """Generate comprehensive Railway readiness report"""
    print("🔍 RAILWAY READINESS CHECK")
    print("=" * 50)
    
    all_issues = []
    all_warnings = []
    
    # Run all checks
    checks = [
        check_file_sizes,
        check_railway_config,
        check_environment_variables,
        check_dependencies,
        check_api_security,
        check_railway_optimization
    ]
    
    for check in checks:
        issues, warnings = check()
        all_issues.extend(issues)
        all_warnings.extend(warnings)
    
    # Summary
    print("\n📋 SUMMARY")
    print("=" * 50)
    
    if not all_issues and not all_warnings:
        print("🎉 EXCELLENT! Your project is Railway-ready!")
        return True
    
    if all_issues:
        print(f"❌ CRITICAL ISSUES ({len(all_issues)}):")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
        print()
    
    if all_warnings:
        print(f"⚠️  WARNINGS ({len(all_warnings)}):")
        for i, warning in enumerate(all_warnings, 1):
            print(f"   {i}. {warning}")
        print()
    
    # Railway deployment readiness
    if not all_issues:
        print("✅ RAILWAY DEPLOYMENT: READY")
        print("   Your app should deploy successfully on Railway")
    else:
        print("❌ RAILWAY DEPLOYMENT: NEEDS FIXES")
        print("   Please address critical issues before deploying")
    
    return len(all_issues) == 0

def main():
    """Main execution"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    railway_ready = generate_report()
    
    print("\n🚀 NEXT STEPS:")
    if railway_ready:
        print("1. Commit your changes: git add . && git commit -m 'Railway optimization'")
        print("2. Push to Railway: git push")
        print("3. Monitor deployment logs")
        print("4. Test your API endpoints")
    else:
        print("1. Fix critical issues listed above")
        print("2. Re-run this checker")
        print("3. Deploy when all critical issues are resolved")
    
    return 0 if railway_ready else 1

if __name__ == "__main__":
    sys.exit(main())