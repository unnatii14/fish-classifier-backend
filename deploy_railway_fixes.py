#!/usr/bin/env python3
"""
Quick commit and push script for Railway deployment fixes
"""
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False
    return True

def deploy_fixes():
    """Deploy Railway fixes"""
    print("🚀 Deploying Railway Health Check Fixes")
    print("=" * 50)
    
    # Git status
    if not run_command("git status", "Checking git status"):
        return
    
    # Add files
    if not run_command("git add .", "Adding all files"):
        return
    
    # Commit
    commit_msg = "Fix Railway deployment health checks and optimize for memory constraints"
    if not run_command(f'git commit -m "{commit_msg}"', "Committing changes"):
        return
    
    # Push
    if not run_command("git push", "Pushing to Railway"):
        return
    
    print("\n" + "=" * 50)
    print("✅ Railway fixes deployed!")
    print("\n📋 Next steps:")
    print("1. Monitor Railway deployment logs")
    print("2. Check health status at your-app.railway.app/health")
    print("3. Test API at your-app.railway.app/")
    print("4. Run: python test_railway_optimized.py https://your-app.railway.app")

if __name__ == "__main__":
    deploy_fixes()