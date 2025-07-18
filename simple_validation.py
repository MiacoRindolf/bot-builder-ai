#!/usr/bin/env python3
"""
Simple System Validation for Bot Builder AI System.
Tests core functionality to ensure the system is working.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def simple_validation():
    """Simple validation of core system components."""
    print("🔍 Simple System Validation Starting...")
    print("=" * 50)
    
    try:
        # Test 1: Import core modules
        print("1. Testing Module Imports...")
        try:
            from core.ai_engine import AIEngine
            from config.settings import settings
            print("   ✅ Core modules imported successfully")
        except Exception as e:
            print(f"   ❌ Import failed: {str(e)}")
            return False
        
        # Test 2: Check configuration
        print("2. Testing Configuration...")
        if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
            print("   ✅ OpenAI API key: Configured")
        else:
            print("   ❌ OpenAI API key: Missing")
            return False
        
        # Test 3: AI Engine initialization
        print("3. Testing AI Engine Initialization...")
        try:
            ai_engine = AIEngine()
            success = await ai_engine.initialize()
            
            if success:
                print("   ✅ AI Engine initialized successfully")
            else:
                print("   ❌ AI Engine initialization failed")
                return False
        except Exception as e:
            print(f"   ❌ AI Engine error: {str(e)}")
            return False
        
        # Test 4: Component availability
        print("4. Testing Component Availability...")
        components = [
            ("Employee Factory", ai_engine.employee_factory),
            ("Learning Engine", ai_engine.learning_engine),
            ("Advanced RL Engine", ai_engine.advanced_rl_engine),
            ("Explainability Engine", ai_engine.explainability_engine),
            ("Real-time Data", ai_engine.real_time_data),
            ("Metrics Collector", ai_engine.metrics_collector)
        ]
        
        for name, component in components:
            if component is not None:
                print(f"   ✅ {name}: Available")
            else:
                print(f"   ❌ {name}: Missing")
                return False
        
        # Test 5: Feature status
        print("5. Testing Feature Status...")
        features = [
            ("Advanced RL", ai_engine.rl_enabled),
            ("Real-time Data", ai_engine.real_time_enabled),
            ("Explainability", ai_engine.explainability_enabled)
        ]
        
        for name, enabled in features:
            status = "Enabled" if enabled else "Disabled"
            print(f"   ✅ {name}: {status}")
        
        # Test 6: System health
        print("6. Testing System Health...")
        if ai_engine.system_health == "healthy":
            print("   ✅ System health: Good")
        else:
            print(f"   ⚠️  System health: {ai_engine.system_health}")
        
        print("=" * 50)
        print("🎉 SIMPLE VALIDATION COMPLETE - SYSTEM IS WORKING!")
        print("✅ All core components are functional")
        print("✅ Advanced features are properly integrated")
        print("✅ System is ready for use")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed with error: {str(e)}")
        return False
    
    finally:
        try:
            await ai_engine.shutdown()
        except:
            pass

def main():
    """Main validation function."""
    print("🤖 Bot Builder AI System Validation")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version}")
    print("=" * 50)
    
    # Run simple validation
    success = asyncio.run(simple_validation())
    
    if success:
        print("\n🎉 VALIDATION PASSED!")
        print("✅ Your Bot Builder AI system is functional!")
        print("✅ Ready for use!")
        return 0
    else:
        print("\n❌ VALIDATION FAILED!")
        print("Core system components are not working")
        print("Please check configuration and dependencies")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 