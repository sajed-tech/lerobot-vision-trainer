#!/usr/bin/env python3
"""
Startup script for Robot Vision Trainer
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

def check_requirements():
    """Check if requirements are installed."""
    try:
        import flask
        import torch
        import requests
        import dotenv
        return True
    except ImportError:
        return False

def install_requirements():
    """Install required packages."""
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
        return True
    else:
        print("Error: requirements.txt not found")
        return False

def check_env_file():
    """Check if .env file exists, create from example if not."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print("No .env file found. Creating from example...")
        example_content = """# Robot Vision Trainer - Combined Service Configuration

# API keys for object detection
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Model configuration for object detection
VISION_MODEL=llama-3.2-11b-vision-instruct:free
DEFAULT_PROMPT="Analyze this image and identify objects that a robot could interact with. List every visible object and suggest possible actions for each object. Format your response in JSON with objects as keys and actions as values. Example: {\\\"cup\\\": [\\\"pick up\\\", \\\"pour liquid\\\"], \\\"door\\\": [\\\"open\\\", \\\"close\\\"]}."

# Hugging Face token for accessing datasets
HF_TOKEN=your_huggingface_token_here

# Weights & Biases configuration
USE_WANDB=false
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=robot_vision_trainer

# Storage directories
UPLOAD_FOLDER=static/uploads
MODELS_DIR=models
TRAINING_HISTORY_FILE=training_history.json

# Flask configuration
FLASK_ENV=development
PORT=5000
HOST=0.0.0.0
"""
        with open(env_path, "w") as f:
            f.write(example_content)
        print(f".env file created at {env_path}")
        print("Please edit this file with your API keys before running the application.")
        return False
    return True

def create_directories():
    """Create required directories if they don't exist."""
    base_dir = Path(__file__).parent
    directories = [
        base_dir / "static" / "uploads",
        base_dir / "models"
    ]
    
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True)
            print(f"Created directory: {directory}")

def main():
    """Main function to start the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Automated Imitation Learning Pipeline')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    args = parser.parse_args()
    
    # Set environment variables for port and host
    os.environ["PORT"] = str(args.port)
    os.environ["HOST"] = args.host
    
    # Check and create directories
    create_directories()
    
    # Check if requirements are installed
    if not check_requirements():
        print("Some required packages are missing. Installing...")
        if not install_requirements():
            print("Failed to install requirements. Please install them manually:")
            print("pip install -r requirements.txt")
            return
    
    # Check if .env file exists
    if not check_env_file():
        return
    
    # Start the application
    print(f"Starting Automated Imitation Learning Pipeline on port {args.port}...")
    subprocess.run([sys.executable, Path(__file__).parent / "app.py"])

if __name__ == "__main__":
    main() 