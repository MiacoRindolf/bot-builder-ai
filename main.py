"""
Main entry point for the Bot Builder AI system.
Initializes and runs the complete AI Employee management system.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings, validate_configuration
from core.ai_engine import AIEngine
from monitoring.metrics import MetricsCollector
from data.data_manager import DataManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BotBuilderAI:
    """
    Main Bot Builder AI application class.
    
    This class orchestrates the entire AI Employee management system,
    including initialization, monitoring, and graceful shutdown.
    """
    
    def __init__(self):
        """Initialize the Bot Builder AI system."""
        self.ai_engine = None
        self.metrics_collector = None
        self.data_manager = None
        self.is_running = False
        
        logger.info("Bot Builder AI system initializing...")
    
    async def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Validate configuration
            config_issues = validate_configuration()
            if config_issues:
                logger.error("Configuration validation failed:")
                for issue in config_issues:
                    logger.error(f"  - {issue}")
                return False
            
            logger.info("Configuration validation passed")
            
            # Initialize data manager
            logger.info("Initializing Data Manager...")
            self.data_manager = DataManager()
            
            # Initialize metrics collector
            logger.info("Initializing Metrics Collector...")
            self.metrics_collector = MetricsCollector()
            await self.metrics_collector.start_monitoring()
            
            # Initialize AI Engine
            logger.info("Initializing AI Engine...")
            self.ai_engine = AIEngine()
            
            logger.info("Bot Builder AI system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            return False
    
    async def start(self):
        """Start the Bot Builder AI system."""
        try:
            if not await self.initialize():
                logger.error("Failed to initialize Bot Builder AI system")
                return False
            
            self.is_running = True
            logger.info("Bot Builder AI system started successfully")
            
            # Keep the system running
            while self.is_running:
                await asyncio.sleep(1)
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            return await self.shutdown()
        except Exception as e:
            logger.error(f"Error during system operation: {str(e)}")
            return await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the Bot Builder AI system gracefully."""
        try:
            logger.info("Shutting down Bot Builder AI system...")
            
            self.is_running = False
            
            # Shutdown AI Engine
            if self.ai_engine:
                await self.ai_engine.shutdown()
                logger.info("AI Engine shutdown complete")
            
            # Shutdown metrics collector
            if self.metrics_collector:
                await self.metrics_collector.stop_monitoring()
                logger.info("Metrics Collector shutdown complete")
            
            logger.info("Bot Builder AI system shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current system status.
        
        Returns:
            Dictionary with system status information
        """
        try:
            status = {
                "system": "Bot Builder AI",
                "version": "1.0.0",
                "status": "running" if self.is_running else "stopped",
                "components": {}
            }
            
            # AI Engine status
            if self.ai_engine:
                status["components"]["ai_engine"] = {
                    "status": "active",
                    "active_employees": len(self.ai_engine.active_ai_employees),
                    "conversations": len(self.ai_engine.conversations)
                }
            else:
                status["components"]["ai_engine"] = {"status": "inactive"}
            
            # Metrics Collector status
            if self.metrics_collector:
                status["components"]["metrics_collector"] = {
                    "status": "active" if self.metrics_collector.is_monitoring else "inactive",
                    "monitoring": self.metrics_collector.is_monitoring
                }
            else:
                status["components"]["metrics_collector"] = {"status": "inactive"}
            
            # Data Manager status
            if self.data_manager:
                status["components"]["data_manager"] = {
                    "status": "active",
                    "cache_size": len(self.data_manager.cache)
                }
            else:
                status["components"]["data_manager"] = {"status": "inactive"}
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {"error": str(e)}

async def main():
    """Main function to run the Bot Builder AI system."""
    try:
        # Create and start the system
        bot_builder = BotBuilderAI()
        
        # Start the system
        success = await bot_builder.start()
        
        if success:
            logger.info("Bot Builder AI system completed successfully")
        else:
            logger.error("Bot Builder AI system failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        sys.exit(1)

def run_streamlit():
    """Run the Streamlit web interface."""
    try:
        import subprocess
        import sys
        
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "ui/streamlit_app.py", 
            "--server.port", str(settings.streamlit_port),
            "--server.address", "0.0.0.0"
        ])
        
    except Exception as e:
        logger.error(f"Error running Streamlit: {str(e)}")
        sys.exit(1)

def run_gradio():
    """Run the Gradio web interface."""
    try:
        import subprocess
        import sys
        
        # Run Gradio
        subprocess.run([
            sys.executable, "ui/gradio_app.py"
        ])
        
    except Exception as e:
        logger.error(f"Error running Gradio: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bot Builder AI System")
    parser.add_argument(
        "--mode", 
        choices=["api", "streamlit", "gradio"], 
        default="api",
        help="Run mode: api (backend only), streamlit (web UI), or gradio (alternative UI)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    if args.mode == "api":
        # Run the API/backend system
        asyncio.run(main())
    elif args.mode == "streamlit":
        # Run Streamlit web interface
        run_streamlit()
    elif args.mode == "gradio":
        # Run Gradio web interface
        run_gradio()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1) 