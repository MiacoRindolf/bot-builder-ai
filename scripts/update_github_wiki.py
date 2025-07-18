#!/usr/bin/env python3
"""
Update GitHub Wiki with latest self-improvement version documentation.

Usage:
  python scripts/update_github_wiki.py

Environment variables required:
  GITHUB_WIKI_URL   - The HTTPS URL of the GitHub wiki repo (e.g. https://github.com/youruser/yourrepo.wiki.git)
  GITHUB_TOKEN      - (Optional) GitHub token for authentication (if private)

This script will:
- Export version history, changelog, and upgrade guides from VersionTracker in Markdown
- Clone or pull the wiki repo to ./wiki
- Write Home.md, Changelog.md, Upgrade-Guide.md, Release-Notes.md
- Commit and push changes
- Print the wiki URL at the end
"""

import os
import sys
import subprocess
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.version_tracker import VersionTracker

WIKI_DIR = project_root / "wiki"

async def export_markdown():
    vt = VersionTracker()
    await vt.initialize()
    # Export markdown for each page
    changelog = await vt.export_version_history(format="markdown")
    upgrade_guide = vt._generate_upgrade_guide()
    release_notes = vt._generate_release_notes()
    # Home page summary
    history = await vt.get_upgrade_history()
    home_md = f"""# Bot Builder AI - Wiki

Welcome to the Bot Builder AI Wiki!

**Current Version:** {history.current_version}
**Total Upgrades:** {history.total_upgrades}
**Total Improvements:** {history.total_improvements:.1f}%
**Success Rate:** {history.success_rate:.1f}%

See [Changelog](Changelog.md), [Upgrade Guide](Upgrade-Guide.md), and [Release Notes](Release-Notes.md) for details.
"""
    return {
        "Home.md": home_md,
        "Changelog.md": changelog,
        "Upgrade-Guide.md": upgrade_guide,
        "Release-Notes.md": release_notes
    }

def run_cmd(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout

def ensure_wiki_repo():
    wiki_url = os.environ.get("GITHUB_WIKI_URL")
    if not wiki_url:
        print("ERROR: GITHUB_WIKI_URL environment variable not set.")
        sys.exit(1)
    if not WIKI_DIR.exists():
        print(f"Cloning wiki repo from {wiki_url}...")
        run_cmd(["git", "clone", wiki_url, str(WIKI_DIR)])
    else:
        print("Pulling latest changes from wiki repo...")
        run_cmd(["git", "pull"], cwd=WIKI_DIR)

def update_wiki_files(markdown_pages):
    for filename, content in markdown_pages.items():
        with open(WIKI_DIR / filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {filename}")

def commit_and_push():
    run_cmd(["git", "add", "-A"], cwd=WIKI_DIR)
    run_cmd(["git", "commit", "-m", "Update wiki with latest self-improvement version docs"], cwd=WIKI_DIR)
    run_cmd(["git", "push"], cwd=WIKI_DIR)

def main():
    print("=== Bot Builder AI: Update GitHub Wiki ===")
    ensure_wiki_repo()
    markdown_pages = asyncio.run(export_markdown())
    update_wiki_files(markdown_pages)
    try:
        commit_and_push()
        print("✅ Wiki updated and pushed!")
    except RuntimeError as e:
        if "nothing to commit" in str(e):
            print("No changes to commit.")
        else:
            raise
    wiki_url = os.environ.get("GITHUB_WIKI_URL", "").replace(".git", "")
    print(f"\nCheck your wiki at: {wiki_url}")

if __name__ == "__main__":
    main() 