#!/usr/bin/env python3
"""
Wiki Content Copier
Helps copy wiki documentation content to clipboard for easy pasting into GitHub Wiki
"""

import pyperclip
import sys
from pathlib import Path

def copy_file_to_clipboard(filename):
    """Copy a specific file's content to clipboard"""
    wiki_dir = Path("wiki_docs")
    file_path = wiki_dir / filename
    
    if not file_path.exists():
        print(f"❌ File not found: {filename}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pyperclip.copy(content)
        print(f"✅ Copied {filename} to clipboard!")
        print(f"📝 Content length: {len(content)} characters")
        return True
        
    except Exception as e:
        print(f"❌ Error copying {filename}: {e}")
        return False

def list_files():
    """List all available wiki files"""
    wiki_dir = Path("wiki_docs")
    
    if not wiki_dir.exists():
        print("❌ Wiki docs directory not found!")
        return
    
    files = list(wiki_dir.glob("*.md"))
    print("📖 Available wiki files:")
    for i, file in enumerate(files, 1):
        print(f"   {i}. {file.name}")
    
    return files

def main():
    """Main function"""
    print("📋 Wiki Content Copier")
    print("=" * 30)
    
    files = list_files()
    if not files:
        return
    
    print("\n🔗 GitHub Wiki Upload Instructions:")
    print("1. Go to: https://github.com/MiacoRindolf/bot-builder-ai/wiki")
    print("2. Click 'New Page'")
    print("3. Use the filename as page name (without .md)")
    print("4. Paste the copied content")
    print("5. Click 'Save Page'")
    
    print("\n📋 Copy Options:")
    print("1. Copy Home.md")
    print("2. Copy Upgrade-History.md")
    print("3. Copy Self-Improvement-Guide.md")
    print("4. Copy API-Documentation.md")
    print("5. List files")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\n❓ Enter your choice (1-6): ").strip()
            
            if choice == "1":
                copy_file_to_clipboard("Home.md")
            elif choice == "2":
                copy_file_to_clipboard("Upgrade-History.md")
            elif choice == "3":
                copy_file_to_clipboard("Self-Improvement-Guide.md")
            elif choice == "4":
                copy_file_to_clipboard("API-Documentation.md")
            elif choice == "5":
                list_files()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        import pyperclip
    except ImportError:
        print("❌ pyperclip not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pyperclip"])
        import pyperclip
    
    main() 