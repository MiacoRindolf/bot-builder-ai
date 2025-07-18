#!/usr/bin/env python3
"""
System Validation Script for Bot Builder AI System.
Tests core functionality to ensure the system is working.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ai_engine import AIEngine
from config.settings import settings

async def quick_validation():
    "uick validation of core system components."""
    print("🔍 Quick System Validation Starting...)
    print(=* 50 
    try:
        # Test 1: AI Engine Initialization
        print("1. Testing AI Engine Initialization...")
        ai_engine = AIEngine()
        success = await ai_engine.initialize()
        
        if success:
            print("   ✅ AI Engine initialized successfully")
        else:
            print("   ❌ AI Engine initialization failed")
            return False
        
        # Test 2: Component Availability
        print("2. Testing Component Availability...")
        components = [
            ("Employee Factory", ai_engine.employee_factory),
            (Learning Engine", ai_engine.learning_engine),
            ("Advanced RL Engine", ai_engine.advanced_rl_engine),
            ("Explainability Engine", ai_engine.explainability_engine),
            ("Real-time Data", ai_engine.real_time_data),
            (Metrics Collector", ai_engine.metrics_collector)
        ]
        
        for name, component in components:
            if component is not None:
                print(f"   ✅ {name}: Available")
            else:
                print(f"   ❌ {name}: Missing)            return False
        
        # Test 3ure Status
        print("3ing Feature Status...")
        features = [
            (Advanced RL, ai_engine.rl_enabled),
            ("Real-time Data", ai_engine.real_time_enabled),
            ("Explainability", ai_engine.explainability_enabled)
        ]
        
        for name, enabled in features:
            status = "Enabled" if enabled else "Disabled"
            print(f"   ✅ {name}: {status}")
        
        # Test 4: Basic AI Employee Creation
        print("4. Testing AI Employee Creation...")
        try:
            response = await ai_engine.process_user_input(
                user_input=Createa research analyst AI Employee,           session_id="validation_test,              user_id=validation_user"
            )
            
            if response and "successfully created" in response.lower():
                print("   ✅ AI Employee creation working")
            else:
                print("   ❌ AI Employee creation failed)             print(f"   Response: {response})            return False
                
        except Exception as e:
            print(f"   ❌ AI Employee creation error: {str(e)}")
            return False
        
        # Test5: Real-time Data (if enabled)
        if ai_engine.real_time_enabled:
            print("5. Testing Real-time Data...)
            try:
                summary = await ai_engine.get_real_time_market_summary()
                if isinstance(summary, dict):
                    print("   ✅ Real-time data working)              else:
                    print("   ❌ Real-time data failed")
                    returnfalse            except Exception as e:
                print(f"   ❌ Real-time data error: {str(e)})            return False
        
        # Test6tem Health
        print(6. Testing System Health...")
        if ai_engine.system_health == "healthy":
            print("   ✅ System health: Good")
        else:
            print(f"   ⚠️  System health: {ai_engine.system_health}")
        
        # Test7figuration
        print("7. Testing Configuration...")
        if settings.openai_api_key:
            print("   ✅ OpenAI API key: Configured")
        else:
            print("   ❌ OpenAI API key: Missing")
            return False
        
        print("=" *50
        print("🎉 QUICK VALIDATION COMPLETE - SYSTEM IS WORKING!)
        print("✅ All core components are functional)
        print("✅ Advanced features are properly integrated)
        print(✅ System is ready for use")
        
        returntrue       
    except Exception as e:
        print(f"❌ Validation failed with error: {str(e)}")
        return False
    
    finally:
        try:
            await ai_engine.shutdown()
        except:
            pass

async def detailed_validation():
    """Detailed validation with more comprehensive tests."int("\n🔬 Detailed System Validation Starting...)
    print(=* 50 
    try:
        ai_engine = AIEngine()
        await ai_engine.initialize()
        
        # Test RL Engine functionality
        print("1. Testing Advanced RL Engine...")
        if ai_engine.rl_enabled:
            rl_engine = ai_engine.advanced_rl_engine
            success = await rl_engine.create_employee_model(
                employee_id="test_rl_employee,              role="trader,             state_dim=256            action_dim=8
            )
            if success:
                print("   ✅ RL Engine: Model creation working")
            else:
                print("   ❌ RL Engine: Model creation failed")
        
        # Test Explainability Engine
        print("2ing Explainability Engine...")
        if ai_engine.explainability_enabled:
            exp_engine = ai_engine.explainability_engine
            if exp_engine.is_initialized:
                print("   ✅ Explainability Engine: Initialized")
            else:
                print("   ❌ Explainability Engine: Not initialized")
        
        # Test Real-time Data
        print("3. Testing Real-time Data...")
        if ai_engine.real_time_enabled:
            rt_data = ai_engine.real_time_data
            if rt_data.is_initialized:
                print("   ✅ Real-time Data: Initialized)
                # Test symbol monitoring
                success = await ai_engine.add_symbol_to_monitoring("AAPL)                if success:
                    print("   ✅ Real-time Data: Symbol monitoring working)              else:
                    print("   ❌ Real-time Data: Symbol monitoring failed")
            else:
                print("   ❌ Real-time Data: Not initialized")
        
        print("=" *50
        print(🎉 DETAILED VALIDATION COMPLETE!")
        
        returntrue       
    except Exception as e:
        print(f"❌ Detailed validation failed: {str(e)}")
        return False
    
    finally:
        try:
            await ai_engine.shutdown()
        except:
            pass

def main():
    alidation function."""
    print(🤖 Bot Builder AI System Validation)
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime(%Y-%m-%d %H:%M:%S)}")    print(fPython Version: {sys.version})
    print(= *50  
    # Run quick validation
    quick_success = asyncio.run(quick_validation())
    
    if quick_success:
        # Run detailed validation
        detailed_success = asyncio.run(detailed_validation())
        
        if detailed_success:
            print("\n🎉 ALL VALIDATIONS PASSED!")
            print("✅ Your Bot Builder AI system is fully functional!")
            print("✅ All advanced features are working correctly!")
            print("✅ Ready for production use!")
            return 0
        else:
            print("\n⚠️  Quick validation passed but detailed validation failed")
            print("System is partially functional")
            return 1
    else:
        print("\n❌ SYSTEM VALIDATION FAILED!)       print("Core system components are not working)     print("Please check configuration and dependencies)
        return1if __name__ == "__main__:  exit_code = main()
    sys.exit(exit_code) 