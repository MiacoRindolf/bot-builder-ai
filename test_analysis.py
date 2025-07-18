#!/usr/bin/env python3
"""
Test script to verify the improved system analysis functionality.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.self_improvement_engine import SelfImprovementEngine
from config.settings import settings

async def test_system_analysis():
    """Test the improved system analysis."""
    print("🧪 Testing Improved System Analysis...")
    
    # Initialize the self-improvement engine
    engine = SelfImprovementEngine()
    await engine.initialize()
    
    # Clear any cached analysis
    engine.cached_analysis = None
    engine.last_analysis = None
    
    print("📊 Running system analysis...")
    
    # Run the analysis
    analysis = await engine.analyze_system()
    
    print(f"\n📈 Analysis Results:")
    print(f"Timestamp: {analysis.timestamp}")
    print(f"Health Score: {analysis.system_health_score}")
    print(f"Performance Metrics: {len(analysis.performance_metrics)} items")
    print(f"Code Quality Metrics: {len(analysis.code_quality_metrics)} items")
    print(f"Technical Debt: {len(analysis.technical_debt_analysis)} items")
    
    print(f"\n🔍 Identified Issues ({len(analysis.identified_issues)}):")
    for i, issue in enumerate(analysis.identified_issues, 1):
        print(f"  {i}. {issue.get('type', 'unknown')}: {issue.get('title', issue.get('description', 'No title'))}")
    
    print(f"\n💡 Improvement Opportunities ({len(analysis.improvement_opportunities)}):")
    for i, opp in enumerate(analysis.improvement_opportunities, 1):
        print(f"  {i}. {opp.get('title', 'No title')}")
        print(f"     Type: {opp.get('type', 'unknown')}")
        print(f"     Priority: {opp.get('priority', 'unknown')}")
        print(f"     Risk: {opp.get('risk_level', 'unknown')}")
    
    # Check if we're getting intelligent analysis or still the old fallback
    if any(issue.get('type') == 'json_parsing_error' for issue in analysis.identified_issues):
        print("\n❌ Still getting old fallback analysis with JSON parsing error")
        return False
    else:
        print("\n✅ Getting intelligent analysis based on actual metrics!")
        return True

if __name__ == "__main__":
    success = asyncio.run(test_system_analysis())
    if success:
        print("\n🎉 System analysis is working correctly!")
    else:
        print("\n⚠️  System analysis still has issues")
    sys.exit(0 if success else 1) 