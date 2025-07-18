#!/usr/bin/env python3
"""
Startup script for the Bot Builder AI Gradio interface.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the Gradio app."""
    try:
        logger.info("Starting Bot Builder AI Gradio Interface on http://localhost:7861...")
        
        # Import and run the Gradio app
        from ui.gradio_app import main as gradio_main
        gradio_main()
        
    except KeyboardInterrupt:
        logger.info("Gradio app stopped by user")
    except Exception as e:
        logger.error(f"Error starting Gradio app: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 