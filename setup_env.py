#!/usr/bin/env python3
"""
Environment setup script for Bot Builder AI.
Handles the creation of .env file from the configuration template.
"""

import os
import shutil
from pathlib import Path

def setup_environment():
    """Setup the environment configuration."""
    print("🔧 Setting up Bot Builder AI environment...")
    
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Source and destination files
    source_file = current_dir / "environment_config.txt"
    dest_file = current_dir / ".env"
    
    try:
        # Check if source file exists
        if not source_file.exists():
            print("❌ Error: environment_config.txt not found!")
            print("   Please ensure the file exists in the current directory.")
            return False
        
        # Copy the configuration to .env
        shutil.copy2(source_file, dest_file)
        
        print("✅ Environment configuration copied successfully!")
        print(f"   Source: {source_file}")
        print(f"   Destination: {dest_file}")
        
        # Verify the file was created
        if dest_file.exists():
            print("✅ .env file created successfully!")
            
            # Read and verify the OpenAI key is present
            with open(dest_file, 'r') as f:
                content = f.read()
                if 'OPENAI_API_KEY=sk-proj-' in content:
                    print("✅ OpenAI API key found in configuration!")
                else:
                    print("⚠️  Warning: OpenAI API key not found in configuration!")
            
            return True
        else:
            print("❌ Error: Failed to create .env file!")
            return False
            
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        return False

def verify_setup():
    """Verify the setup is correct."""
    print("\n🔍 Verifying setup...")
    
    current_dir = Path(__file__).parent
    env_file = current_dir / ".env"
    
    if not env_file.exists():
        print("❌ .env file not found!")
        return False
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
            
        # Check for required configurations
        required_keys = [
            'OPENAI_API_KEY=',
            'DATABASE_URL=',
            'LOG_LEVEL='
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in content:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ Missing required configurations: {missing_keys}")
            return False
        
        print("✅ All required configurations found!")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying setup: {str(e)}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating necessary directories...")
    
    current_dir = Path(__file__).parent
    directories = [
        "logs",
        "data/storage", 
        "monitoring/metrics_data",
        "data/cache"
    ]
    
    for dir_path in directories:
        full_path = current_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def main():
    """Main setup function."""
    print("🤖 Bot Builder AI - Environment Setup")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Setup failed! Please check the error messages above.")
        return
    
    # Create directories
    create_directories()
    
    # Verify setup
    if not verify_setup():
        print("\n❌ Setup verification failed! Please check the configuration.")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the system: python start.py")
    print("3. Or run directly: streamlit run ui/streamlit_app.py")
    print("\n🔗 The system will be available at:")
    print("   - Streamlit: http://localhost:8501")
    print("   - Gradio: http://localhost:7860")
    
    print("\n⚠️  Important notes:")
    print("- Your OpenAI API key is now configured")
    print("- The .env file contains sensitive information - keep it secure")
    print("- For production, change the default security keys")
    print("- Monitor your OpenAI API usage to avoid unexpected charges")

if __name__ == "__main__":
    main() 