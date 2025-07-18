#!/usr/bin/env python3
"""
Quick Test for Bot Builder AI Wiki System
Simple tests to verify basic functionality
"""

import asyncio
import sys
from pathlib import Path
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Run quick tests"""
    print("🚀 Quick Wiki System Test")
    print("=" * 40)
    
    # Test 1: Check if wiki generator exists
    try:
        from utils.wiki_generator import WikiGenerator
        print("✅ Wiki Generator import successful")
    except Exception as e:
        print(f"❌ Wiki Generator import failed: {e}")
        return False
    
    # Test 2: Check if upload helper exists
    try:
        from upload_wiki import main as upload_main
        print("✅ Upload Helper import successful")
    except Exception as e:
        print(f"❌ Upload Helper import failed: {e}")
        return False
    
    # Test 3: Check if wiki docs directory exists
    wiki_dir = Path("wiki_docs")
    if wiki_dir.exists():
        print("✅ Wiki docs directory exists")
        
        # Check for files
        files = list(wiki_dir.glob("*.md"))
        print(f"✅ Found {len(files)} markdown files:")
        for file in files:
            print(f"   - {file.name}")
    else:
        print("❌ Wiki docs directory not found")
        return False
    
    # Test 4: Check file contents
    home_file = wiki_dir / "Home.md"
    if home_file.exists():
        with open(home_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "Bot Builder AI System" in content:
            print("✅ Home.md contains expected content")
        else:
            print("❌ Home.md missing expected content")
            return False
    else:
        print("❌ Home.md not found")
        return False
    
    print("\n🎉 Quick test completed successfully!")
    return True

async def test_wiki_generation():
    """Test wiki generation functionality"""
    print("\n📝 Testing Wiki Generation...")
    
    try:
        from utils.wiki_generator import WikiGenerator
        
        generator = WikiGenerator()
        docs = await generator.generate_documentation()
        
        print(f"✅ Generated {len(docs)} documentation files")
        
        # Check file names
        expected_files = ['Home.md', 'Upgrade-History.md', 'Self-Improvement-Guide.md', 'API-Documentation.md']
        for expected_file in expected_files:
            if any(doc['filename'] == expected_file for doc in docs):
                print(f"✅ {expected_file} generated")
            else:
                print(f"❌ {expected_file} missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Wiki generation failed: {e}")
        return False

def main():
    """Main test runner"""
    print("🤖 Bot Builder AI - Quick Wiki Test")
    print("=" * 50)
    
    # Run quick tests
    if not quick_test():
        print("\n❌ Quick test failed!")
        return
    
    # Run wiki generation test
    asyncio.run(test_wiki_generation())
    
    print("\n✅ All quick tests passed!")
    print("\n📋 Next Steps:")
    print("1. Run comprehensive test: python test_wiki_system.py")
    print("2. Upload to GitHub: python upload_wiki.py")
    print("3. View generated docs in: wiki_docs/")

if __name__ == "__main__":
    main() 