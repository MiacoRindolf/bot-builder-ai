"""
Version Tracker - Self-Improvement Change Documentation
Tracks all self-improvement changes, versions, and their impact on the system.
"""

import asyncio
import json
import logging
import os
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Types of changes that can be made."""
    BUG_FIX = "bug_fix"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    FEATURE_ADDITION = "feature_addition"
    CODE_REFACTORING = "code_refactoring"
    SECURITY_UPDATE = "security_update"
    DOCUMENTATION_UPDATE = "documentation_update"
    ARCHITECTURE_CHANGE = "architecture_change"
    OPTIMIZATION = "optimization"

class ImpactLevel(Enum):
    """Impact levels of changes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class VersionChange:
    """Represents a single change in a version."""
    id: str
    change_type: ChangeType
    title: str
    description: str
    files_modified: List[str]
    lines_added: int
    lines_removed: int
    impact_level: ImpactLevel
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    improvement_percentage: float
    implemented_by: str
    approved_by: str
    timestamp: datetime
    rollback_available: bool = True
    rollback_reason: Optional[str] = None

@dataclass
class SystemVersion:
    """Represents a complete system version."""
    version: str
    major: int
    minor: int
    patch: int
    build: int
    release_date: datetime
    changes: List[VersionChange]
    total_changes: int
    performance_improvement: float
    bug_fixes: int
    new_features: int
    breaking_changes: int
    release_notes: str
    upgrade_guide: str
    rollback_instructions: str
    compatibility_notes: str

@dataclass
class UpgradeHistory:
    """Complete upgrade history of the system."""
    current_version: str
    versions: List[SystemVersion]
    total_upgrades: int
    total_improvements: float
    last_upgrade_date: datetime
    upgrade_frequency_days: float
    success_rate: float
    average_improvement_per_upgrade: float

class VersionTracker:
    """
    Version Tracker for Self-Improvement Changes.
    
    Responsibilities:
    - Track all self-improvement changes
    - Maintain version history
    - Document impact and improvements
    - Generate upgrade reports
    - Provide rollback capabilities
    """
    
    def __init__(self):
        """Initialize the Version Tracker."""
        self.version_file = Path("version_history.json")
        self.changes_file = Path("changes_log.json")
        self.current_version = "1.0.0"
        self.version_history: List[SystemVersion] = []
        self.pending_changes: List[VersionChange] = []
        
        # Version tracking
        self.major_version = 1
        self.minor_version = 0
        self.patch_version = 0
        self.build_number = 0
        
        # Metrics tracking
        self.baseline_metrics: Dict[str, Any] = {}
        self.current_metrics: Dict[str, Any] = {}
        
        logger.info("Version Tracker initialized successfully")
    
    async def initialize(self) -> bool:
        """Initialize the Version Tracker."""
        try:
            # Load existing version history
            await self._load_version_history()
            
            # Set baseline metrics
            await self._capture_baseline_metrics()
            
            logger.info("Version Tracker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Version Tracker: {str(e)}")
            return False
    
    async def record_improvement(self, proposal_id: str, change_type: ChangeType, 
                               title: str, description: str, files_modified: List[str],
                               lines_added: int, lines_removed: int, impact_level: ImpactLevel,
                               implemented_by: str, approved_by: str) -> str:
        """
        Record a self-improvement change.
        
        Args:
            proposal_id: ID of the improvement proposal
            change_type: Type of change made
            title: Title of the change
            description: Detailed description
            files_modified: List of files modified
            lines_added: Number of lines added
            lines_removed: Number of lines removed
            impact_level: Impact level of the change
            implemented_by: Who implemented the change
            approved_by: Who approved the change
            
        Returns:
            Change ID
        """
        try:
            # Capture metrics before change
            before_metrics = await self._capture_current_metrics()
            
            # Create change record
            change = VersionChange(
                id=proposal_id,
                change_type=change_type,
                title=title,
                description=description,
                files_modified=files_modified,
                lines_added=lines_added,
                lines_removed=lines_removed,
                impact_level=impact_level,
                before_metrics=before_metrics,
                after_metrics={},  # Will be updated after implementation
                improvement_percentage=0.0,  # Will be calculated after implementation
                implemented_by=implemented_by,
                approved_by=approved_by,
                timestamp=datetime.now()
            )
            
            # Add to pending changes
            self.pending_changes.append(change)
            
            # Save to file
            await self._save_changes_log()
            
            logger.info(f"Recorded improvement change: {change.id}")
            return change.id
            
        except Exception as e:
            logger.error(f"Error recording improvement: {str(e)}")
            return ""
    
    async def finalize_improvement(self, change_id: str, success: bool = True, 
                                 rollback_reason: Optional[str] = None) -> bool:
        """
        Finalize an improvement change with post-implementation metrics.
        
        Args:
            change_id: ID of the change to finalize
            success: Whether the implementation was successful
            rollback_reason: Reason for rollback if applicable
            
        Returns:
            True if finalized successfully, False otherwise
        """
        try:
            # Find the change
            change = None
            for pending_change in self.pending_changes:
                if pending_change.id == change_id:
                    change = pending_change
                    break
            
            if not change:
                logger.error(f"Change {change_id} not found in pending changes")
                return False
            
            if success:
                # Capture metrics after change
                after_metrics = await self._capture_current_metrics()
                change.after_metrics = after_metrics
                
                # Calculate improvement percentage
                improvement = self._calculate_improvement_percentage(
                    change.before_metrics, after_metrics
                )
                change.improvement_percentage = improvement
                
                # Move to version history
                await self._add_change_to_version(change)
                
                logger.info(f"Finalized improvement {change_id} with {improvement:.1f}% improvement")
            else:
                # Mark as rolled back
                change.rollback_available = False
                change.rollback_reason = rollback_reason
                
                # Keep in pending changes for audit trail
                logger.info(f"Marked improvement {change_id} as rolled back")
            
            # Save changes
            await self._save_changes_log()
            await self._save_version_history()
            
            return True
            
        except Exception as e:
            logger.error(f"Error finalizing improvement: {str(e)}")
            return False
    
    async def create_new_version(self, version_type: str = "patch") -> str:
        """
        Create a new version from pending changes.
        
        Args:
            version_type: Type of version bump ("major", "minor", "patch")
            
        Returns:
            New version string
        """
        try:
            if not self.pending_changes:
                logger.warning("No pending changes to create new version")
                return self.current_version
            
            # Bump version number
            if version_type == "major":
                self.major_version += 1
                self.minor_version = 0
                self.patch_version = 0
            elif version_type == "minor":
                self.minor_version += 1
                self.patch_version = 0
            else:  # patch
                self.patch_version += 1
            
            self.build_number += 1
            
            new_version = f"{self.major_version}.{self.minor_version}.{self.patch_version}"
            
            # Create system version
            system_version = SystemVersion(
                version=new_version,
                major=self.major_version,
                minor=self.minor_version,
                patch=self.patch_version,
                build=self.build_number,
                release_date=datetime.now(),
                changes=self.pending_changes.copy(),
                total_changes=len(self.pending_changes),
                performance_improvement=self._calculate_total_improvement(),
                bug_fixes=len([c for c in self.pending_changes if c.change_type == ChangeType.BUG_FIX]),
                new_features=len([c for c in self.pending_changes if c.change_type == ChangeType.FEATURE_ADDITION]),
                breaking_changes=len([c for c in self.pending_changes if c.impact_level == ImpactLevel.CRITICAL]),
                release_notes=self._generate_release_notes(),
                upgrade_guide=self._generate_upgrade_guide(),
                rollback_instructions=self._generate_rollback_instructions(),
                compatibility_notes=self._generate_compatibility_notes()
            )
            
            # Add to version history
            self.version_history.append(system_version)
            self.current_version = new_version
            
            # Clear pending changes
            self.pending_changes.clear()
            
            # Save version history
            await self._save_version_history()
            
            logger.info(f"Created new version: {new_version}")
            return new_version
            
        except Exception as e:
            logger.error(f"Error creating new version: {str(e)}")
            return self.current_version
    
    async def get_upgrade_history(self) -> UpgradeHistory:
        """Get complete upgrade history."""
        try:
            if not self.version_history:
                return UpgradeHistory(
                    current_version=self.current_version,
                    versions=[],
                    total_upgrades=0,
                    total_improvements=0.0,
                    last_upgrade_date=datetime.now(),
                    upgrade_frequency_days=0.0,
                    success_rate=100.0,
                    average_improvement_per_upgrade=0.0
                )
            
            # Calculate statistics
            total_upgrades = len(self.version_history)
            total_improvements = sum(v.performance_improvement for v in self.version_history)
            
            # Calculate upgrade frequency
            if total_upgrades > 1:
                first_upgrade = self.version_history[0].release_date
                last_upgrade = self.version_history[-1].release_date
                days_between = (last_upgrade - first_upgrade).days
                upgrade_frequency = days_between / (total_upgrades - 1)
            else:
                upgrade_frequency = 0.0
            
            # Calculate success rate
            successful_upgrades = len([v for v in self.version_history if v.performance_improvement > 0])
            success_rate = (successful_upgrades / total_upgrades) * 100
            
            # Calculate average improvement
            average_improvement = total_improvements / total_upgrades
            
            return UpgradeHistory(
                current_version=self.current_version,
                versions=self.version_history,
                total_upgrades=total_upgrades,
                total_improvements=total_improvements,
                last_upgrade_date=self.version_history[-1].release_date if self.version_history else datetime.now(),
                upgrade_frequency_days=upgrade_frequency,
                success_rate=success_rate,
                average_improvement_per_upgrade=average_improvement
            )
            
        except Exception as e:
            logger.error(f"Error getting upgrade history: {str(e)}")
            return UpgradeHistory(
                current_version=self.current_version,
                versions=[],
                total_upgrades=0,
                total_improvements=0.0,
                last_upgrade_date=datetime.now(),
                upgrade_frequency_days=0.0,
                success_rate=0.0,
                average_improvement_per_upgrade=0.0
            )
    
    async def get_version_report(self, version: str) -> Optional[SystemVersion]:
        """Get detailed report for a specific version."""
        try:
            for v in self.version_history:
                if v.version == version:
                    return v
            return None
            
        except Exception as e:
            logger.error(f"Error getting version report: {str(e)}")
            return None
    
    async def get_recent_changes(self, days: int = 30) -> List[VersionChange]:
        """Get recent changes within specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_changes = []
            
            for version in self.version_history:
                if version.release_date >= cutoff_date:
                    recent_changes.extend(version.changes)
            
            return recent_changes
            
        except Exception as e:
            logger.error(f"Error getting recent changes: {str(e)}")
            return []
    
    async def export_version_history(self, format: str = "json") -> str:
        """Export version history in specified format."""
        try:
            history = await self.get_upgrade_history()
            
            if format.lower() == "json":
                return json.dumps(asdict(history), indent=2, default=str)
            elif format.lower() == "markdown":
                return self._generate_markdown_report(history)
            else:
                return json.dumps(asdict(history), indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error exporting version history: {str(e)}")
            return ""
    
    async def _load_version_history(self):
        """Load version history from file."""
        try:
            if self.version_file.exists():
                with open(self.version_file, 'r') as f:
                    data = json.load(f)
                    self.version_history = [SystemVersion(**v) for v in data.get("versions", [])]
                    self.current_version = data.get("current_version", "1.0.0")
                    
                    # Parse version numbers
                    version_parts = self.current_version.split(".")
                    if len(version_parts) >= 3:
                        self.major_version = int(version_parts[0])
                        self.minor_version = int(version_parts[1])
                        self.patch_version = int(version_parts[2])
                        self.build_number = int(version_parts[3]) if len(version_parts) > 3 else 0
                    
                    logger.info(f"Loaded version history: {len(self.version_history)} versions")
                    
        except Exception as e:
            logger.error(f"Error loading version history: {str(e)}")
    
    async def _save_version_history(self):
        """Save version history to file."""
        try:
            data = {
                "current_version": self.current_version,
                "versions": [asdict(v) for v in self.version_history]
            }
            
            with open(self.version_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving version history: {str(e)}")
    
    async def _save_changes_log(self):
        """Save changes log to file."""
        try:
            data = {
                "pending_changes": [asdict(c) for c in self.pending_changes]
            }
            
            with open(self.changes_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving changes log: {str(e)}")
    
    async def _capture_baseline_metrics(self):
        """Capture baseline system metrics."""
        try:
            # This would capture actual system metrics
            # For now, use placeholder metrics
            self.baseline_metrics = {
                "performance_score": 0.8,
                "code_quality": 0.75,
                "test_coverage": 0.85,
                "response_time": 1.2,
                "error_rate": 0.05,
                "memory_usage": 0.6,
                "cpu_usage": 0.4
            }
            
        except Exception as e:
            logger.error(f"Error capturing baseline metrics: {str(e)}")
    
    async def _capture_current_metrics(self) -> Dict[str, Any]:
        """Capture current system metrics."""
        try:
            # This would capture actual current metrics
            # For now, use placeholder metrics with some variation
            import random
            
            return {
                "performance_score": 0.8 + random.uniform(-0.1, 0.1),
                "code_quality": 0.75 + random.uniform(-0.05, 0.05),
                "test_coverage": 0.85 + random.uniform(-0.02, 0.02),
                "response_time": 1.2 + random.uniform(-0.1, 0.1),
                "error_rate": 0.05 + random.uniform(-0.01, 0.01),
                "memory_usage": 0.6 + random.uniform(-0.05, 0.05),
                "cpu_usage": 0.4 + random.uniform(-0.05, 0.05)
            }
            
        except Exception as e:
            logger.error(f"Error capturing current metrics: {str(e)}")
            return {}
    
    def _calculate_improvement_percentage(self, before: Dict[str, Any], after: Dict[str, Any]) -> float:
        """Calculate improvement percentage between before and after metrics."""
        try:
            if not before or not after:
                return 0.0
            
            improvements = []
            
            # Performance metrics (higher is better)
            for metric in ["performance_score", "code_quality", "test_coverage"]:
                if metric in before and metric in after:
                    if before[metric] > 0:
                        improvement = ((after[metric] - before[metric]) / before[metric]) * 100
                        improvements.append(improvement)
            
            # Response time (lower is better)
            if "response_time" in before and "response_time" in after:
                if before["response_time"] > 0:
                    improvement = ((before["response_time"] - after["response_time"]) / before["response_time"]) * 100
                    improvements.append(improvement)
            
            # Error rate (lower is better)
            if "error_rate" in before and "error_rate" in after:
                if before["error_rate"] > 0:
                    improvement = ((before["error_rate"] - after["error_rate"]) / before["error_rate"]) * 100
                    improvements.append(improvement)
            
            return sum(improvements) / len(improvements) if improvements else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating improvement percentage: {str(e)}")
            return 0.0
    
    def _calculate_total_improvement(self) -> float:
        """Calculate total improvement from pending changes."""
        try:
            if not self.pending_changes:
                return 0.0
            
            total_improvement = sum(c.improvement_percentage for c in self.pending_changes)
            return total_improvement
            
        except Exception as e:
            logger.error(f"Error calculating total improvement: {str(e)}")
            return 0.0
    
    async def _add_change_to_version(self, change: VersionChange):
        """Add a change to the current version."""
        try:
            # This would add the change to the current version
            # For now, just log it
            logger.info(f"Added change {change.id} to version {self.current_version}")
            
        except Exception as e:
            logger.error(f"Error adding change to version: {str(e)}")
    
    def _generate_release_notes(self) -> str:
        """Generate release notes for the current version."""
        try:
            if not self.pending_changes:
                return "No changes in this version."
            
            notes = f"# Version {self.current_version} Release Notes\n\n"
            notes += f"**Release Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            notes += f"**Total Changes:** {len(self.pending_changes)}\n\n"
            
            # Group changes by type
            changes_by_type = {}
            for change in self.pending_changes:
                if change.change_type.value not in changes_by_type:
                    changes_by_type[change.change_type.value] = []
                changes_by_type[change.change_type.value].append(change)
            
            for change_type, changes in changes_by_type.items():
                notes += f"## {change_type.replace('_', ' ').title()}\n\n"
                for change in changes:
                    notes += f"- **{change.title}** ({change.impact_level.value})\n"
                    notes += f"  {change.description}\n\n"
            
            return notes
            
        except Exception as e:
            logger.error(f"Error generating release notes: {str(e)}")
            return "Error generating release notes."
    
    def _generate_upgrade_guide(self) -> str:
        """Generate upgrade guide for the current version."""
        try:
            if not self.pending_changes:
                return "No upgrade required."
            
            guide = f"# Upgrade Guide for Version {self.current_version}\n\n"
            guide += "## Overview\n\n"
            guide += f"This version includes {len(self.pending_changes)} improvements with an overall performance improvement of {self._calculate_total_improvement():.1f}%.\n\n"
            
            guide += "## Breaking Changes\n\n"
            breaking_changes = [c for c in self.pending_changes if c.impact_level == ImpactLevel.CRITICAL]
            if breaking_changes:
                for change in breaking_changes:
                    guide += f"- **{change.title}**: {change.description}\n"
            else:
                guide += "No breaking changes in this version.\n"
            
            guide += "\n## Upgrade Steps\n\n"
            guide += "1. Backup your current system\n"
            guide += "2. Review the changes in this version\n"
            guide += "3. Test in a staging environment\n"
            guide += "4. Deploy to production\n"
            guide += "5. Monitor system performance\n"
            
            return guide
            
        except Exception as e:
            logger.error(f"Error generating upgrade guide: {str(e)}")
            return "Error generating upgrade guide."
    
    def _generate_rollback_instructions(self) -> str:
        """Generate rollback instructions for the current version."""
        try:
            instructions = f"# Rollback Instructions for Version {self.current_version}\n\n"
            instructions += "## Emergency Rollback\n\n"
            instructions += "If you need to rollback this version:\n\n"
            instructions += "1. Stop the current system\n"
            instructions += "2. Restore from backup\n"
            instructions += "3. Restart the system\n"
            instructions += "4. Verify system functionality\n\n"
            
            instructions += "## Partial Rollback\n\n"
            instructions += "To rollback specific changes:\n\n"
            for change in self.pending_changes:
                if change.rollback_available:
                    instructions += f"- **{change.title}**: {change.description}\n"
                    instructions += f"  Files: {', '.join(change.files_modified)}\n\n"
            
            return instructions
            
        except Exception as e:
            logger.error(f"Error generating rollback instructions: {str(e)}")
            return "Error generating rollback instructions."
    
    def _generate_compatibility_notes(self) -> str:
        """Generate compatibility notes for the current version."""
        try:
            notes = f"# Compatibility Notes for Version {self.current_version}\n\n"
            notes += "## System Requirements\n\n"
            notes += "- Python 3.8+\n"
            notes += "- OpenAI API access\n"
            notes += "- Sufficient memory and CPU resources\n\n"
            
            notes += "## Dependencies\n\n"
            notes += "All dependencies are specified in requirements.txt\n\n"
            
            notes += "## Known Issues\n\n"
            notes += "None reported in this version.\n\n"
            
            notes += "## Future Compatibility\n\n"
            notes += "This version maintains backward compatibility with previous versions.\n"
            
            return notes
            
        except Exception as e:
            logger.error(f"Error generating compatibility notes: {str(e)}")
            return "Error generating compatibility notes."
    
    def _generate_markdown_report(self, history: UpgradeHistory) -> str:
        """Generate markdown report of upgrade history."""
        try:
            report = "# Bot Builder AI - Upgrade History Report\n\n"
            report += f"**Current Version:** {history.current_version}\n"
            report += f"**Total Upgrades:** {history.total_upgrades}\n"
            report += f"**Total Improvements:** {history.total_improvements:.1f}%\n"
            report += f"**Success Rate:** {history.success_rate:.1f}%\n"
            report += f"**Average Improvement per Upgrade:** {history.average_improvement_per_upgrade:.1f}%\n\n"
            
            report += "## Version History\n\n"
            for version in history.versions:
                report += f"### Version {version.version}\n\n"
                report += f"**Release Date:** {version.release_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
                report += f"**Changes:** {version.total_changes}\n"
                report += f"**Performance Improvement:** {version.performance_improvement:.1f}%\n"
                report += f"**Bug Fixes:** {version.bug_fixes}\n"
                report += f"**New Features:** {version.new_features}\n"
                report += f"**Breaking Changes:** {version.breaking_changes}\n\n"
                
                report += "#### Changes:\n\n"
                for change in version.changes:
                    report += f"- **{change.title}** ({change.change_type.value})\n"
                    report += f"  {change.description}\n"
                    report += f"  Impact: {change.impact_level.value}\n"
                    report += f"  Improvement: {change.improvement_percentage:.1f}%\n\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating markdown report: {str(e)}")
            return "Error generating markdown report." 