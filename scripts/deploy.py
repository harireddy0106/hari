#scripts/deploy.py
"""
Deployment script for FloatChat
"""
import subprocess
import sys

def deploy():
    print("🚀 Deploying FloatChat...")
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("✅ Deployment successful!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    deploy()