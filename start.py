#!/usr/bin/env python3
"""
Startup script for the Bot Builder AI system.
Provides easy access to different modes and configurations.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main startup function."""
    print("🤖 Bot Builder AI - Advanced AI Employee Management System")
    print("=" * 60)
    print()
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  Warning: No .env file found!")
        print("   Please copy env_example.txt to .env and configure your settings.")
        print("   At minimum, you need to set your OPENAI_API_KEY.")
        print()
    
    # Show available modes
    print("Available modes:")
    print("1. Streamlit Web Interface (Recommended)")
    print("2. Gradio Web Interface")
    print("3. API/Backend Only")
    print("4. Help")
    print()
    
    try:
        choice = input("Select mode (1-4): ").strip()
        
        if choice == "1":
            print("🚀 Starting Streamlit interface...")
            os.system(f"{sys.executable} -m streamlit run ui/streamlit_app.py --server.port 8501")
        elif choice == "2":
            print("🚀 Starting Gradio interface...")
            os.system(f"{sys.executable} ui/gradio_app.py")
        elif choice == "3":
            print("🚀 Starting API/Backend system...")
            os.system(f"{sys.executable} main.py --mode api")
        elif choice == "4":
            show_help()
        else:
            print("❌ Invalid choice. Please select 1-4.")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def show_help():
    """Show help information."""
    print()
    print("📋 Bot Builder AI - Help")
    print("=" * 30)
    print()
    print("This system allows you to create and manage specialized AI Employees")
    print("for an AI-powered hedge fund.")
    print()
    print("Available AI Employee Roles:")
    print("• Research Analyst: Deep learning, forecasting, economic analysis")
    print("• Trader: Reinforcement learning, execution speed, strategic decision-making")
    print("• Risk Manager: Probability theory, statistical modeling, scenario testing")
    print("• Compliance Officer: Regulatory knowledge, NLP, explainability")
    print("• Data Specialist: Data cleaning, management, structuring")
    print()
    print("Setup Instructions:")
    print("1. Copy env_example.txt to .env")
    print("2. Add your OpenAI API key to .env")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run this script and select your preferred interface")
    print()
    print("For more information, see README.md")

if __name__ == "__main__":
    main() 