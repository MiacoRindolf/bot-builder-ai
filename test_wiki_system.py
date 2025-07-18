#!/usr/bin/env python3
"""
Test Script for Bot Builder AI Wiki Documentation System
Tests all components: wiki generator, version tracker integration, and documentation quality
"""

import asyncio
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.wiki_generator import WikiGenerator
from core.version_tracker import VersionTracker, ChangeType, ImpactLevel

class WikiSystemTester:
    """Comprehensive tester for the wiki documentation system"""
    
    def __init__(self):
        self.test_results = []
        self.wiki_generator = WikiGenerator()
        self.version_tracker = VersionTracker()
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log a test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        print(f"{status} {test_name}")
        if details:
            print(f"   📝 {details}")
    
    async def test_version_tracker_integration(self):
        """Test Version Tracker integration"""
        print("\n🔍 Testing Version Tracker Integration...")
        
        try:
            # Initialize version tracker
            success = await self.version_tracker.initialize()
            self.log_test("Version Tracker Initialization", success)
            
            # Test getting upgrade history
            upgrade_history = await self.version_tracker.get_upgrade_history()
            self.log_test("Get Upgrade History", upgrade_history is not None)
            
            # Test version info access
            if upgrade_history:
                self.log_test("Current Version Access", hasattr(upgrade_history, 'current_version'))
                self.log_test("Total Upgrades Access", hasattr(upgrade_history, 'total_upgrades'))
                self.log_test("Success Rate Access", hasattr(upgrade_history, 'success_rate'))
                
                print(f"   📊 Current Version: {upgrade_history.current_version}")
                print(f"   📊 Total Upgrades: {upgrade_history.total_upgrades}")
                print(f"   📊 Success Rate: {upgrade_history.success_rate:.1f}%")
            
        except Exception as e:
            self.log_test("Version Tracker Integration", False, f"Error: {str(e)}")
    
    async def test_wiki_generation(self):
        """Test wiki documentation generation"""
        print("\n📝 Testing Wiki Documentation Generation...")
        
        try:
            # Generate documentation
            docs = await self.wiki_generator.generate_documentation()
            
            # Test number of files generated
            expected_files = 4
            self.log_test("Documentation File Count", len(docs) == expected_files, 
                         f"Generated {len(docs)} files (expected {expected_files})")
            
            # Test each file
            expected_files = ['Home.md', 'Upgrade-History.md', 'Self-Improvement-Guide.md', 'API-Documentation.md']
            for expected_file in expected_files:
                file_found = any(doc['filename'] == expected_file for doc in docs)
                self.log_test(f"File Generation: {expected_file}", file_found)
            
            # Test file content quality
            for doc in docs:
                content = doc['content']
                
                # Test content length
                min_length = 1000  # Minimum content length
                self.log_test(f"Content Length: {doc['filename']}", 
                             len(content) >= min_length,
                             f"Length: {len(content)} chars (min: {min_length})")
                
                # Test for required sections
                required_sections = ['# ', '## ', '### ']
                sections_found = sum(1 for section in required_sections if section in content)
                self.log_test(f"Content Structure: {doc['filename']}", 
                             sections_found >= 2,
                             f"Found {sections_found} section levels")
                
                # Test for dynamic content (version info, etc.)
                if 'Upgrade-History' in doc['filename']:
                    has_version_info = any(keyword in content for keyword in 
                                          ['Current Version', 'Total Upgrades', 'Success Rate'])
                    self.log_test(f"Dynamic Content: {doc['filename']}", has_version_info)
            
        except Exception as e:
            self.log_test("Wiki Generation", False, f"Error: {str(e)}")
    
    def test_file_creation(self):
        """Test that files were actually created on disk"""
        print("\n💾 Testing File Creation...")
        
        wiki_dir = Path("wiki_docs")
        
        # Test directory exists
        self.log_test("Wiki Directory Creation", wiki_dir.exists())
        
        if wiki_dir.exists():
            # Test each expected file
            expected_files = ['Home.md', 'Upgrade-History.md', 'Self-Improvement-Guide.md', 'API-Documentation.md']
            for expected_file in expected_files:
                file_path = wiki_dir / expected_file
                self.log_test(f"File Exists: {expected_file}", file_path.exists())
                
                if file_path.exists():
                    # Test file size
                    file_size = file_path.stat().st_size
                    self.log_test(f"File Size: {expected_file}", file_size > 0,
                                 f"Size: {file_size} bytes")
                    
                    # Test file is readable
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        self.log_test(f"File Readable: {expected_file}", len(content) > 0)
                    except Exception as e:
                        self.log_test(f"File Readable: {expected_file}", False, f"Error: {str(e)}")
    
    def test_documentation_quality(self):
        """Test the quality and completeness of generated documentation"""
        print("\n📊 Testing Documentation Quality...")
        
        wiki_dir = Path("wiki_docs")
        
        if not wiki_dir.exists():
            self.log_test("Documentation Quality", False, "Wiki directory not found")
            return
        
        # Test Home.md quality
        home_file = wiki_dir / "Home.md"
        if home_file.exists():
            with open(home_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Test for key sections
            key_sections = [
                'Bot Builder AI System',
                'Key Features',
                'System Architecture',
                'Quick Start',
                'Performance Metrics'
            ]
            
            sections_found = sum(1 for section in key_sections if section in content)
            self.log_test("Home.md Key Sections", sections_found >= 4,
                         f"Found {sections_found}/{len(key_sections)} key sections")
            
            # Test for links and navigation
            has_links = '[' in content and '](' in content
            self.log_test("Home.md Navigation Links", has_links)
        
        # Test API Documentation quality
        api_file = wiki_dir / "API-Documentation.md"
        if api_file.exists():
            with open(api_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Test for API endpoints
            api_sections = [
                'GET /api/status',
                'POST /api/employees/create',
                'GET /api/self-improvement/status',
                'Authentication',
                'Rate Limits'
            ]
            
            api_sections_found = sum(1 for section in api_sections if section in content)
            self.log_test("API Documentation Completeness", api_sections_found >= 3,
                         f"Found {api_sections_found}/{len(api_sections)} API sections")
        
        # Test Self-Improvement Guide quality
        guide_file = wiki_dir / "Self-Improvement-Guide.md"
        if guide_file.exists():
            with open(guide_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Test for process explanation
            process_sections = [
                'Analysis Phase',
                'Proposal Phase',
                'Implementation Phase',
                'Learning Phase',
                'CEO Approval Workflow'
            ]
            
            process_sections_found = sum(1 for section in process_sections if section in content)
            self.log_test("Self-Improvement Guide Completeness", process_sections_found >= 4,
                         f"Found {process_sections_found}/{len(process_sections)} process sections")
    
    def test_upload_helper(self):
        """Test the upload helper functionality"""
        print("\n📤 Testing Upload Helper...")
        
        try:
            # Import and test upload helper
            from upload_wiki import main as upload_main
            
            # Test that the function exists and is callable
            self.log_test("Upload Helper Import", callable(upload_main))
            
            # Test wiki directory detection
            wiki_dir = Path("wiki_docs")
            self.log_test("Upload Helper Directory Detection", wiki_dir.exists())
            
        except Exception as e:
            self.log_test("Upload Helper", False, f"Error: {str(e)}")
    
    async def run_comprehensive_test(self):
        """Run all tests"""
        print("🤖 Bot Builder AI - Wiki System Comprehensive Test")
        print("=" * 60)
        
        # Run all test suites
        await self.test_version_tracker_integration()
        await self.test_wiki_generation()
        self.test_file_creation()
        self.test_documentation_quality()
        self.test_upload_helper()
        
        # Generate test report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate a comprehensive test report"""
        print("\n📋 Test Report")
        print("=" * 30)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Show failed tests
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['success']:
                    print(f"   - {result['test']}: {result['details']}")
        
        # Overall assessment
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Wiki system is ready for use.")
        elif passed_tests >= total_tests * 0.8:
            print("\n⚠️  MOST TESTS PASSED. Wiki system is mostly functional.")
        else:
            print("\n🚨 MANY TESTS FAILED. Wiki system needs attention.")
        
        # Save detailed report
        report_file = Path("wiki_test_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'success_rate': (passed_tests/total_tests)*100
                },
                'results': self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file.absolute()}")

async def main():
    """Main test runner"""
    tester = WikiSystemTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main()) 