#!/usr/bin/env python3
"""
Wiki Upload Helper for Bot Builder AI System
Provides instructions and tools for uploading documentation to GitHub Wiki
"""

import os
import subprocess
import webbrowser
from pathlib import Path

def main():
    """Main entry point"""
    print("🤖 Bot Builder AI - Wiki Upload Helper")
    print("=" * 50)
    
    wiki_dir = Path("wiki_docs")
    
    if not wiki_dir.exists():
        print("❌ Wiki documentation not found!")
        print("Please run 'python utils/wiki_generator.py' first.")
        return
    
    print("✅ Wiki documentation found!")
    print(f"📁 Location: {wiki_dir.absolute()}")
    
    # List generated files
    print("\n📖 Generated Files:")
    for file in wiki_dir.glob("*.md"):
        print(f"   - {file.name}")
    
    print("\n🚀 Upload Instructions:")
    print("=" * 30)
    
    print("\n1️⃣ **Create GitHub Wiki** (if it doesn't exist):")
    print("   - Go to: https://github.com/MiacoRindolf/bot-builder-ai")
    print("   - Click on 'Wiki' tab")
    print("   - Click 'Create the first page' if wiki doesn't exist")
    
    print("\n2️⃣ **Upload Files Manually**:")
    print("   - For each .md file in the wiki_docs folder:")
    print("     * Click 'New Page' in the wiki")
    print("     * Use the filename as the page name (without .md)")
    print("     * Copy and paste the content from the file")
    print("     * Click 'Save Page'")
    
    print("\n3️⃣ **Alternative: Use GitHub CLI** (if installed):")
    print("   ```bash")
    print("   # Clone the wiki repository")
    print("   gh repo clone MiacoRindolf/bot-builder-ai.wiki")
    print("   cd bot-builder-ai.wiki")
    print("   ")
    print("   # Copy files from wiki_docs")
    print("   cp ../wiki_docs/*.md .")
    print("   ")
    print("   # Commit and push")
    print("   git add .")
    print("   git commit -m 'Add Bot Builder AI documentation'")
    print("   git push")
    print("   ```")
    
    print("\n4️⃣ **Page Structure**:")
    print("   - Home.md → Home page (main entry point)")
    print("   - Upgrade-History.md → Upgrade History page")
    print("   - Self-Improvement-Guide.md → Self-Improvement Guide")
    print("   - API-Documentation.md → API Documentation")
    
    print("\n5️⃣ **Navigation**:")
    print("   - The Home page will be the main entry point")
    print("   - Add navigation links between pages")
    print("   - Use relative links: [[Upgrade-History]]")
    
    # Offer to open the repository
    print("\n🔗 Quick Links:")
    print("   - Repository: https://github.com/MiacoRindolf/bot-builder-ai")
    print("   - Wiki: https://github.com/MiacoRindolf/bot-builder-ai/wiki")
    
    # Ask if user wants to open the repository
    try:
        response = input("\n❓ Would you like to open the GitHub repository? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            webbrowser.open("https://github.com/MiacoRindolf/bot-builder-ai")
            print("✅ Opened GitHub repository in browser")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    print("\n✅ Wiki upload instructions complete!")
    print("📖 Your documentation is ready to be uploaded to GitHub Wiki.")

if __name__ == "__main__":
    main() 