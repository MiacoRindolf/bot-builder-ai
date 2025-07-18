"""
Code Generation Engine - AI-Powered Code Synthesis and Modification
Handles code diff generation, patch application, and code synthesis using OpenAI.
"""

import asyncio
import json
import logging
import os
import difflib
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import hashlib

from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class CodeChange:
    """Represents a code change to be applied."""
    file_path: str
    original_content: str
    new_content: str
    diff: str
    change_type: str  # "ADD", "MODIFY", "DELETE"
    line_numbers: Tuple[int, int]  # start_line, end_line
    description: str

@dataclass
class CodeGenerationResult:
    """Result of code generation operation."""
    success: bool
    changes: List[CodeChange]
    error_message: Optional[str] = None
    warnings: List[str] = None
    test_results: Optional[Dict[str, Any]] = None

class CodeGenerationEngine:
    """
    AI-Powered Code Generation Engine.
    
    Responsibilities:
    - Generate code diffs using OpenAI
    - Apply code changes safely
    - Validate code changes
    - Handle rollbacks
    - Code synthesis and refactoring
    """
    
    def __init__(self):
        """Initialize the Code Generation Engine."""
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.project_root = Path.cwd()
        self.backup_dir = self.project_root / "backups" / "code_changes"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Safety settings
        self.max_file_size_mb = 10  # Maximum file size to modify
        self.allowed_file_extensions = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp"}
        self.excluded_directories = {".git", "__pycache__", "node_modules", "venv", "env", ".pytest_cache"}
        
        # Change tracking
        self.applied_changes: List[CodeChange] = []
        self.change_history: List[Dict[str, Any]] = []
        
        logger.info("Code Generation Engine initialized successfully")
    
    async def initialize(self) -> bool:
        """Initialize the Code Generation Engine."""
        try:
            # Validate project structure
            if not self._validate_project_structure():
                logger.error("Invalid project structure")
                return False
            
            logger.info("Code Generation Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Code Generation Engine: {str(e)}")
            return False
    
    async def generate_code_changes(self, target_files: List[str], description: str, 
                                  rationale: str) -> Dict[str, str]:
        """
        Generate code changes for the specified files.
        
        Args:
            target_files: List of file paths to modify
            description: Description of the improvement
            rationale: Rationale for the changes
            
        Returns:
            Dictionary mapping file paths to diffs
        """
        try:
            logger.info(f"Generating code changes for {len(target_files)} files")
            
            code_changes = {}
            
            for file_path in target_files:
                if not self._is_file_safe_to_modify(file_path):
                    logger.warning(f"Skipping unsafe file: {file_path}")
                    continue
                
                # Read current file content
                current_content = self._read_file_content(file_path)
                if current_content is None:
                    continue
                
                # Generate new content using OpenAI
                new_content = await self._generate_new_content(
                    file_path, current_content, description, rationale
                )
                
                if new_content and new_content != current_content:
                    # Generate diff
                    diff = self._generate_diff(current_content, new_content)
                    code_changes[file_path] = diff
                    
                    logger.info(f"Generated changes for {file_path}")
                else:
                    logger.info(f"No changes needed for {file_path}")
            
            return code_changes
            
        except Exception as e:
            logger.error(f"Error generating code changes: {str(e)}")
            return {}
    
    async def apply_code_changes(self, code_changes: Dict[str, str]) -> Dict[str, Any]:
        """
        Apply code changes to the filesystem.
        
        Args:
            code_changes: Dictionary mapping file paths to diffs
            
        Returns:
            Result of the operation
        """
        try:
            logger.info(f"Applying {len(code_changes)} code changes")
            
            # Create backup
            backup_id = self._create_backup()
            
            applied_changes = []
            errors = []
            
            for file_path, diff in code_changes.items():
                try:
                    # Apply the diff
                    success = await self._apply_diff_to_file(file_path, diff)
                    
                    if success:
                        applied_changes.append(file_path)
                        logger.info(f"Successfully applied changes to {file_path}")
                    else:
                        errors.append(f"Failed to apply changes to {file_path}")
                        
                except Exception as e:
                    errors.append(f"Error applying changes to {file_path}: {str(e)}")
            
            # Record the change
            change_record = {
                "timestamp": datetime.now().isoformat(),
                "backup_id": backup_id,
                "applied_changes": applied_changes,
                "errors": errors,
                "total_files": len(code_changes)
            }
            self.change_history.append(change_record)
            
            return {
                "success": len(errors) == 0,
                "applied_changes": applied_changes,
                "errors": errors,
                "backup_id": backup_id,
                "total_files": len(code_changes)
            }
            
        except Exception as e:
            logger.error(f"Error applying code changes: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "applied_changes": [],
                "errors": [str(e)]
            }
    
    async def rollback_changes(self, backup_id: str) -> bool:
        """
        Rollback changes using a backup.
        
        Args:
            backup_id: ID of the backup to restore
            
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            logger.info(f"Rolling back changes using backup: {backup_id}")
            
            backup_path = self.backup_dir / backup_id
            if not backup_path.exists():
                logger.error(f"Backup {backup_id} not found")
                return False
            
            # Restore files from backup
            for file_backup in backup_path.glob("*.py"):
                original_path = self.project_root / file_backup.name
                shutil.copy2(file_backup, original_path)
                logger.info(f"Restored {original_path}")
            
            logger.info(f"Successfully rolled back changes from backup {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back changes: {str(e)}")
            return False
    
    async def validate_code_changes(self, code_changes: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate code changes before applying them.
        
        Args:
            code_changes: Dictionary mapping file paths to diffs
            
        Returns:
            Validation result
        """
        try:
            validation_results = {
                "valid": True,
                "warnings": [],
                "errors": [],
                "file_analysis": {}
            }
            
            for file_path, diff in code_changes.items():
                file_analysis = await self._analyze_file_changes(file_path, diff)
                validation_results["file_analysis"][file_path] = file_analysis
                
                if file_analysis["has_errors"]:
                    validation_results["valid"] = False
                    validation_results["errors"].extend(file_analysis["errors"])
                
                if file_analysis["has_warnings"]:
                    validation_results["warnings"].extend(file_analysis["warnings"])
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating code changes: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "warnings": [],
                "errors": [str(e)]
            }
    
    def get_change_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent change history."""
        return self.change_history[-limit:] if self.change_history else []
    
    def _validate_project_structure(self) -> bool:
        """Validate that we're in a proper project structure."""
        try:
            # Check if we're in a project root (has common project files)
            project_indicators = ["requirements.txt", "setup.py", "pyproject.toml", "package.json", "README.md"]
            has_indicators = any((self.project_root / indicator).exists() for indicator in project_indicators)
            
            if not has_indicators:
                logger.warning("No project indicators found - ensure this is a project root")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating project structure: {str(e)}")
            return False
    
    def _is_file_safe_to_modify(self, file_path: str) -> bool:
        """Check if a file is safe to modify."""
        try:
            path = Path(file_path)
            
            # Check file extension
            if path.suffix not in self.allowed_file_extensions:
                return False
            
            # Check if file exists
            if not path.exists():
                return False
            
            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return False
            
            # Check if in excluded directory
            for excluded in self.excluded_directories:
                if excluded in path.parts:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking file safety: {str(e)}")
            return False
    
    def _read_file_content(self, file_path: str) -> Optional[str]:
        """Read file content safely."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
    async def _generate_new_content(self, file_path: str, current_content: str, 
                                  description: str, rationale: str) -> Optional[str]:
        """Generate new content using OpenAI."""
        try:
            # Create context for the AI
            context = self._create_code_context(file_path, current_content)
            
            prompt = f"""
You are an expert software engineer. Based on the following improvement request, generate improved code for the file.

IMPROVEMENT DESCRIPTION:
{description}

RATIONALE:
{rationale}

FILE: {file_path}

CURRENT CODE:
```python
{current_content}
```

CONTEXT:
{context}

Generate the complete improved code. Maintain the same structure and style, but implement the requested improvements.
Return only the code, no explanations.

IMPROVED CODE:
```python
"""
            
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert Python developer. Generate improved code based on the requirements."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.2
            )
            
            new_content = response.choices[0].message.content.strip()
            
            # Clean up the response (remove markdown if present)
            if new_content.startswith("```python"):
                new_content = new_content[9:]
            if new_content.endswith("```"):
                new_content = new_content[:-3]
            
            return new_content.strip()
            
        except Exception as e:
            logger.error(f"Error generating new content for {file_path}: {str(e)}")
            return None
    
    def _create_code_context(self, file_path: str, content: str) -> str:
        """Create context information for code generation."""
        try:
            # Analyze the code structure
            lines = content.split('\n')
            imports = [line for line in lines if line.strip().startswith(('import ', 'from '))]
            functions = [line for line in lines if line.strip().startswith('def ')]
            classes = [line for line in lines if line.strip().startswith('class ')]
            
            context = f"""
File Analysis:
- Total lines: {len(lines)}
- Import statements: {len(imports)}
- Functions: {len(functions)}
- Classes: {len(classes)}

Key imports:
{chr(10).join(imports[:5])}

Key functions:
{chr(10).join(functions[:5])}

Key classes:
{chr(10).join(classes[:5])}
"""
            return context
            
        except Exception as e:
            logger.error(f"Error creating code context: {str(e)}")
            return "Context analysis failed"
    
    def _generate_diff(self, original_content: str, new_content: str) -> str:
        """Generate a diff between original and new content."""
        try:
            original_lines = original_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            
            diff = difflib.unified_diff(
                original_lines,
                new_lines,
                fromfile='original',
                tofile='modified',
                lineterm=''
            )
            
            return '\n'.join(diff)
            
        except Exception as e:
            logger.error(f"Error generating diff: {str(e)}")
            return ""
    
    async def _apply_diff_to_file(self, file_path: str, diff: str) -> bool:
        """Apply a diff to a file."""
        try:
            # Parse the diff and apply it
            # This is a simplified implementation - in production, you'd use a proper diff library
            path = Path(file_path)
            
            # For now, we'll apply the changes by replacing the entire file
            # In a real implementation, you'd parse the diff and apply it line by line
            new_content = self._apply_unified_diff(path.read_text(), diff)
            
            if new_content is not None:
                path.write_text(new_content, encoding='utf-8')
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error applying diff to {file_path}: {str(e)}")
            return False
    
    def _apply_unified_diff(self, original_content: str, diff: str) -> Optional[str]:
        """Apply a unified diff to content."""
        try:
            # This is a simplified diff application
            # In production, use a proper diff library like patch
            lines = diff.split('\n')
            
            # Find the new content in the diff
            new_content_lines = []
            in_new_content = False
            
            for line in lines:
                if line.startswith('+') and not line.startswith('+++'):
                    new_content_lines.append(line[1:])
                elif line.startswith(' ') and in_new_content:
                    new_content_lines.append(line[1:])
                elif line.startswith('@@'):
                    in_new_content = True
            
            if new_content_lines:
                return '\n'.join(new_content_lines)
            else:
                return original_content
                
        except Exception as e:
            logger.error(f"Error applying unified diff: {str(e)}")
            return None
    
    def _create_backup(self) -> str:
        """Create a backup of current state."""
        try:
            backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
            backup_path = self.backup_dir / backup_id
            backup_path.mkdir(exist_ok=True)
            
            # Backup Python files
            for py_file in self.project_root.rglob("*.py"):
                if self._is_file_safe_to_modify(str(py_file)):
                    relative_path = py_file.relative_to(self.project_root)
                    backup_file = backup_path / relative_path.name
                    shutil.copy2(py_file, backup_file)
            
            logger.info(f"Created backup: {backup_id}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return f"backup_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def _analyze_file_changes(self, file_path: str, diff: str) -> Dict[str, Any]:
        """Analyze changes to a specific file."""
        try:
            analysis = {
                "file_path": file_path,
                "has_errors": False,
                "has_warnings": False,
                "errors": [],
                "warnings": [],
                "change_summary": "",
                "risk_level": "LOW"
            }
            
            # Basic analysis
            lines_added = len([line for line in diff.split('\n') if line.startswith('+') and not line.startswith('+++')])
            lines_removed = len([line for line in diff.split('\n') if line.startswith('-') and not line.startswith('---')])
            
            analysis["change_summary"] = f"Added {lines_added} lines, removed {lines_removed} lines"
            
            # Risk assessment
            if lines_added + lines_removed > 100:
                analysis["risk_level"] = "HIGH"
                analysis["has_warnings"] = True
                analysis["warnings"].append("Large number of changes detected")
            elif lines_added + lines_removed > 50:
                analysis["risk_level"] = "MEDIUM"
                analysis["has_warnings"] = True
                analysis["warnings"].append("Moderate number of changes detected")
            
            # Check for potential issues
            if "import" in diff.lower() and "from" in diff.lower():
                analysis["has_warnings"] = True
                analysis["warnings"].append("Import changes detected - verify dependencies")
            
            if "def " in diff.lower() or "class " in diff.lower():
                analysis["has_warnings"] = True
                analysis["warnings"].append("Function/class changes detected - verify interfaces")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing file changes: {str(e)}")
            return {
                "file_path": file_path,
                "has_errors": True,
                "has_warnings": False,
                "errors": [str(e)],
                "warnings": [],
                "change_summary": "Analysis failed",
                "risk_level": "HIGH"
            } 