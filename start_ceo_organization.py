#!/usr/bin/env python3
"""
CEO AI Organization Startup Script
Comprehensive initialization and validation system.
"""

import asyncio
import logging
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import platform
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/startup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class CEOOrganizationStartup:
    """
    CEO AI Organization Startup and Validation System.
    
    Handles:
    - Environment validation
    - Dependency checks
    - System initialization
    - Health monitoring
    - Graceful startup/shutdown
    """
    
    def __init__(self):
        """Initialize the startup system."""
        self.startup_time = datetime.now()
        self.validation_results = {}
        self.system_ready = False
        
        # Create necessary directories
        self._create_directories()
        
        logger.info("CEO Organization Startup system initialized")
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            'logs',
            'data/cache',
            'data/storage',
            'monitoring/metrics_data',
            'config'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def run_startup_sequence(self) -> bool:
        """Run the complete startup sequence."""
        try:
            logger.info("🚀 Starting CEO-Centric AI Organization System...")
            logger.info("=" * 60)
            
            # Phase 1: Environment Validation
            logger.info("📋 Phase 1: Environment Validation")
            if not await self._validate_environment():
                return False
            
            # Phase 2: Dependency Checks
            logger.info("🔍 Phase 2: Dependency Validation")
            if not await self._validate_dependencies():
                return False
            
            # Phase 3: Configuration Setup
            logger.info("⚙️ Phase 3: Configuration Setup")
            if not await self._setup_configuration():
                return False
            
            # Phase 4: System Initialization
            logger.info("🎯 Phase 4: System Initialization")
            if not await self._initialize_systems():
                return False
            
            # Phase 5: Health Checks
            logger.info("🏥 Phase 5: Health Validation")
            if not await self._validate_system_health():
                return False
            
            # Phase 6: Demo Setup
            logger.info("🎬 Phase 6: Demo Environment Setup")
            await self._setup_demo_environment()
            
            self.system_ready = True
            logger.info("=" * 60)
            logger.info("✅ CEO AI Organization System ready!")
            logger.info(f"🕐 Total startup time: {(datetime.now() - self.startup_time).total_seconds():.2f}s")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {str(e)}")
            return False
    
    async def _validate_environment(self) -> bool:
        """Validate the environment setup."""
        try:
            validations = {
                "python_version": self._check_python_version(),
                "platform": self._check_platform(),
                "memory": self._check_memory(),
                "disk_space": self._check_disk_space(),
                "network": await self._check_network()
            }
            
            self.validation_results["environment"] = validations
            
            # Check if all validations passed
            all_passed = all(validations.values())
            
            for check, result in validations.items():
                status = "✅" if result else "❌"
                logger.info(f"  {status} {check.replace('_', ' ').title()}")
            
            if not all_passed:
                logger.error("❌ Environment validation failed")
                return False
            
            logger.info("✅ Environment validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment validation error: {str(e)}")
            return False
    
    def _check_python_version(self) -> bool:
        """Check Python version."""
        version = sys.version_info
        required = (3, 8)
        
        if version >= required:
            logger.info(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            logger.error(f"  ❌ Python {version.major}.{version.minor} (requires >= {required[0]}.{required[1]})")
            return False
    
    def _check_platform(self) -> bool:
        """Check platform compatibility."""
        system = platform.system()
        logger.info(f"  ✅ Platform: {system} {platform.release()}")
        return True  # We support all platforms
    
    def _check_memory(self) -> bool:
        """Check available memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb >= 4.0:  # 4GB minimum
                logger.info(f"  ✅ Available memory: {available_gb:.1f}GB")
                return True
            else:
                logger.error(f"  ❌ Insufficient memory: {available_gb:.1f}GB (requires >= 4GB)")
                return False
                
        except ImportError:
            logger.warning("  ⚠️ Could not check memory (psutil not available)")
            return True
    
    def _check_disk_space(self) -> bool:
        """Check disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)
            
            if free_gb >= 2.0:  # 2GB minimum
                logger.info(f"  ✅ Free disk space: {free_gb:.1f}GB")
                return True
            else:
                logger.error(f"  ❌ Insufficient disk space: {free_gb:.1f}GB (requires >= 2GB)")
                return False
                
        except Exception:
            logger.warning("  ⚠️ Could not check disk space")
            return True
    
    async def _check_network(self) -> bool:
        """Check network connectivity."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.openai.com/v1/models', timeout=10) as response:
                    if response.status == 401:  # Unauthorized is expected without API key
                        logger.info("  ✅ Network connectivity to OpenAI")
                        return True
                    else:
                        logger.warning(f"  ⚠️ Unexpected response from OpenAI: {response.status}")
                        return True  # Still consider it working
                        
        except Exception as e:
            logger.error(f"  ❌ Network connectivity failed: {str(e)}")
            return False
    
    async def _validate_dependencies(self) -> bool:
        """Validate required dependencies."""
        try:
            critical_packages = [
                'openai',
                'streamlit', 
                'fastapi',
                'pandas',
                'numpy',
                'asyncio'
            ]
            
            optional_packages = [
                'torch',
                'plotly',
                'yfinance',
                'aiohttp'
            ]
            
            # Check critical packages
            missing_critical = []
            for package in critical_packages:
                try:
                    __import__(package)
                    logger.info(f"  ✅ {package}")
                except ImportError:
                    logger.error(f"  ❌ {package} (CRITICAL)")
                    missing_critical.append(package)
            
            # Check optional packages
            missing_optional = []
            for package in optional_packages:
                try:
                    __import__(package)
                    logger.info(f"  ✅ {package}")
                except ImportError:
                    logger.warning(f"  ⚠️ {package} (OPTIONAL)")
                    missing_optional.append(package)
            
            if missing_critical:
                logger.error(f"❌ Missing critical dependencies: {missing_critical}")
                logger.error("Run: pip install -r requirements.txt")
                return False
            
            if missing_optional:
                logger.warning(f"⚠️ Missing optional dependencies: {missing_optional}")
                logger.warning("Some features may be limited")
            
            logger.info("✅ Dependency validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency validation error: {str(e)}")
            return False
    
    async def _setup_configuration(self) -> bool:
        """Setup configuration files."""
        try:
            # Check for .env file
            env_file = Path('.env')
            if not env_file.exists():
                logger.warning("  ⚠️ .env file not found, using defaults")
                
                # Create sample .env
                sample_env = """# CEO AI Organization Configuration
OPENAI_API_KEY=your_openai_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
GITHUB_TOKEN=your_github_token_here
GITHUB_OWNER=your_github_username
GITHUB_REPO=bot-builder-ai

# Database Configuration
DATABASE_URL=sqlite:///./data/ceo_organization.db

# Security
SECRET_KEY=your_secret_key_here

# Logging
LOG_LEVEL=INFO

# Ports
STREAMLIT_PORT=8502
API_PORT=8503
"""
                
                env_file.write_text(sample_env)
                logger.info("  ✅ Created sample .env file")
            else:
                logger.info("  ✅ .env file found")
            
            # Validate critical environment variables
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key or openai_key == 'your_openai_api_key_here':
                logger.error("  ❌ OPENAI_API_KEY not configured")
                logger.error("  Please set your OpenAI API key in the .env file")
                return False
            else:
                logger.info("  ✅ OpenAI API key configured")
            
            logger.info("✅ Configuration setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration setup error: {str(e)}")
            return False
    
    async def _initialize_systems(self) -> bool:
        """Initialize all system components."""
        try:
            # Import and initialize the main organization
            from main_ceo_organization import CEOAIOrganization
            
            logger.info("  🏗️ Creating CEO AI Organization...")
            self.organization = CEOAIOrganization()
            
            logger.info("  🚀 Initializing components...")
            success = await self.organization.initialize()
            
            if success:
                logger.info("✅ System initialization completed")
                return True
            else:
                logger.error("❌ System initialization failed")
                return False
            
        except Exception as e:
            logger.error(f"❌ System initialization error: {str(e)}")
            return False
    
    async def _validate_system_health(self) -> bool:
        """Validate system health after initialization."""
        try:
            # Check if organization is properly initialized
            if not hasattr(self, 'organization') or not self.organization.is_initialized:
                logger.error("  ❌ Organization not properly initialized")
                return False
            
            # Get dashboard to test functionality
            dashboard = await self.organization.get_executive_dashboard()
            
            if dashboard and 'system_overview' in dashboard:
                logger.info("  ✅ Executive dashboard functional")
            else:
                logger.error("  ❌ Executive dashboard not working")
                return False
            
            # Test CEO portal
            if self.organization.ceo_portal:
                ceo_dashboard = await self.organization.ceo_portal.get_ceo_dashboard()
                if ceo_dashboard:
                    logger.info("  ✅ CEO portal functional")
                else:
                    logger.error("  ❌ CEO portal not working")
                    return False
            
            # Test SDLC team
            if self.organization.sdlc_team:
                team_count = len(self.organization.sdlc_team.teams)
                bot_count = len(self.organization.sdlc_team.bots)
                logger.info(f"  ✅ SDLC system functional ({team_count} teams, {bot_count} bots)")
            
            # Test coordinator
            if self.organization.coordinator:
                logger.info("  ✅ Cross-team coordinator functional")
            
            logger.info("✅ System health validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ System health validation error: {str(e)}")
            return False
    
    async def _setup_demo_environment(self):
        """Setup demo environment with sample data."""
        try:
            logger.info("  🎬 Setting up demo environment...")
            
            # Demo environment is already set up in the organization initialization
            # Just log the demo data available
            
            if hasattr(self, 'organization'):
                dashboard = await self.organization.get_executive_dashboard()
                
                pending_decisions = dashboard.get('ceo_portal', {}).get('pending_decisions', {})
                decision_count = pending_decisions.get('total_count', 0)
                
                team_status = dashboard.get('ceo_portal', {}).get('team_status', {})
                team_count = len(team_status)
                
                logger.info(f"  ✅ Demo ready: {decision_count} sample decisions, {team_count} teams")
            
            logger.info("✅ Demo environment setup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Demo setup warning: {str(e)}")
    
    def display_startup_summary(self):
        """Display startup summary and access information."""
        print("\n" + "=" * 80)
        print("🎉 CEO-CENTRIC AI ORGANIZATION SYSTEM READY")
        print("=" * 80)
        
        print("\n📊 SYSTEM STATUS:")
        print(f"  • Status: {'🟢 READY' if self.system_ready else '🔴 FAILED'}")
        print(f"  • Startup Time: {(datetime.now() - self.startup_time).total_seconds():.2f}s")
        print(f"  • Platform: {platform.system()} {platform.release()}")
        
        print("\n🌐 ACCESS POINTS:")
        print("  • CEO Streamlit Portal: http://localhost:8502")
        print("  • API Endpoint: http://localhost:8503")
        print("  • Health Check: http://localhost:8503/health")
        print("  • Executive Dashboard: http://localhost:8503/dashboard")
        
        print("\n🎯 QUICK START:")
        print("  1. Open CEO Portal: streamlit run ui/ceo_streamlit_portal.py --server.port 8502")
        print("  2. Or use API: python main_ceo_organization.py")
        print("  3. Or run both: python start_ceo_organization.py --full")
        
        print("\n💼 CEO CAPABILITIES:")
        print("  • Strategic decision queue with intelligent filtering")
        print("  • Autonomous SDLC bot teams with GitHub integration")
        print("  • Real-time hedge fund AI pods coordination")
        print("  • Cross-team synergy identification and execution")
        print("  • Complete transparency with advanced explainability")
        print("  • Executive dashboard with strategic insights")
        
        print("\n📋 SAMPLE COMMANDS:")
        print("  • 'Show me critical decisions' - View pending critical decisions")
        print("  • 'Team status report' - Get comprehensive team overview")
        print("  • 'Identify synergy opportunities' - Find cross-team synergies")
        print("  • 'Market intelligence summary' - Get real-time market data")
        print("  • 'Strategic alignment status' - Check organizational alignment")
        
        if self.system_ready:
            print("\n🚀 Ready for CEO operations! The AI organization awaits your strategic direction.")
        else:
            print("\n❌ System not ready. Check logs for errors.")
        
        print("=" * 80)

async def main():
    """Main startup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CEO AI Organization Startup")
    parser.add_argument('--full', action='store_true', help='Start full system with all interfaces')
    parser.add_argument('--api-only', action='store_true', help='Start API server only')
    parser.add_argument('--ui-only', action='store_true', help='Start Streamlit UI only')
    parser.add_argument('--validate-only', action='store_true', help='Run validation only')
    
    args = parser.parse_args()
    
    # Create startup system
    startup = CEOOrganizationStartup()
    
    # Run startup sequence
    success = await startup.run_startup_sequence()
    
    if not success:
        print("❌ Startup failed. Check logs for details.")
        sys.exit(1)
    
    # Display summary
    startup.display_startup_summary()
    
    # Handle different startup modes
    if args.validate_only:
        print("✅ Validation complete. Exiting.")
        return
    
    elif args.api_only:
        print("🚀 Starting API server only...")
        from main_ceo_organization import main
        main()
    
    elif args.ui_only:
        print("🚀 Starting Streamlit UI only...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "ui/ceo_streamlit_portal.py", 
            "--server.port", "8502"
        ])
    
    elif args.full:
        print("🚀 Starting full system (API + UI)...")
        
        # Start API in background
        import multiprocessing
        from main_ceo_organization import main as api_main
        
        api_process = multiprocessing.Process(target=api_main)
        api_process.start()
        
        # Give API time to start
        time.sleep(3)
        
        # Start Streamlit UI
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "ui/ceo_streamlit_portal.py", 
            "--server.port", "8502"
        ])
        
        # Cleanup
        api_process.terminate()
        api_process.join()
    
    else:
        print("🎯 System ready. Use --full, --api-only, or --ui-only to start services.")

if __name__ == "__main__":
    asyncio.run(main()) 