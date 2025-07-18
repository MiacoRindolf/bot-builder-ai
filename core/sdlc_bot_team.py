"""
SDLC Bot Team - Software Development Lifecycle Autonomous Bot Organization
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import subprocess

from config.settings import settings
from core.ceo_portal import CEOPortal, DecisionCategory, Priority

logger = logging.getLogger(__name__)

class BotRole(Enum):
    """SDLC Bot roles."""
    # Architecture Team
    CHIEF_ARCHITECT = "CHIEF_ARCHITECT"
    SOLUTION_ARCHITECT = "SOLUTION_ARCHITECT"
    SECURITY_ARCHITECT = "SECURITY_ARCHITECT"
    DATA_ARCHITECT = "DATA_ARCHITECT"
    
    # Development Team
    DEV_LEAD = "DEV_LEAD"
    SENIOR_DEV = "SENIOR_DEV"
    FULL_STACK_DEV = "FULL_STACK_DEV"
    BACKEND_DEV = "BACKEND_DEV"
    FRONTEND_DEV = "FRONTEND_DEV"
    
    # Quality Team
    QA_DIRECTOR = "QA_DIRECTOR"
    UI_QA = "UI_QA"
    UX_QA = "UX_QA"
    PERFORMANCE_QA = "PERFORMANCE_QA"
    SECURITY_QA = "SECURITY_QA"
    
    # Data Team
    DBA_LEAD = "DBA_LEAD"
    SENIOR_DBA = "SENIOR_DBA"
    DATA_ENGINEER = "DATA_ENGINEER"
    ANALYTICS_DBA = "ANALYTICS_DBA"
    
    # Management Team
    TECHNICAL_PM = "TECHNICAL_PM"
    SCRUM_MASTER = "SCRUM_MASTER"
    PRODUCT_OWNER = "PRODUCT_OWNER"
    BUSINESS_ANALYST = "BUSINESS_ANALYST"
    TECHNICAL_WRITER = "TECHNICAL_WRITER"

class TaskStatus(Enum):
    """Task status for SDLC workflow."""
    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    REVIEW = "REVIEW"
    TESTING = "TESTING"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class SDLCTask:
    """A task in the SDLC workflow."""
    id: str
    title: str
    description: str
    task_type: str  # "FEATURE", "BUG", "TECH_DEBT", "RESEARCH", "DOCUMENTATION"
    priority: TaskPriority
    status: TaskStatus
    assigned_bot: str
    assigned_team: str
    created_at: datetime
    updated_at: datetime
    due_date: Optional[datetime]
    estimated_hours: int
    actual_hours: int
    dependencies: List[str]
    blocked_by: List[str]
    github_issue_url: Optional[str]
    github_pr_url: Optional[str]
    acceptance_criteria: List[str]
    progress_percentage: int
    context: Dict[str, Any]

@dataclass
class SDLCBot:
    """An autonomous SDLC bot."""
    id: str
    name: str
    role: BotRole
    team: str
    specializations: List[str]
    skills: Dict[str, float]  # Skill name -> proficiency (0-1)
    current_tasks: List[str]
    completed_tasks: int
    success_rate: float
    availability: float  # 0-1, where 1 is fully available
    last_active: datetime
    status: str  # "ACTIVE", "BUSY", "OFFLINE", "MAINTENANCE"
    performance_metrics: Dict[str, Any]
    learning_progress: Dict[str, float]

@dataclass
class SDLCTeam:
    """An SDLC team with lead and members."""
    name: str
    team_type: str
    lead_bot: str
    member_bots: List[str]
    current_sprint: Optional[str]
    active_tasks: List[str]
    completed_tasks: List[str]
    team_metrics: Dict[str, Any]
    objectives: List[str]
    last_standup: Optional[datetime]

class GitHubIntegration:
    """GitHub integration for autonomous progress documentation."""
    
    def __init__(self, repo_owner: str, repo_name: str, token: str):
        """Initialize GitHub integration."""
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        logger.info(f"GitHub integration initialized for {repo_owner}/{repo_name}")
    
    async def create_issue(self, title: str, description: str, labels: List[str] = None) -> Optional[str]:
        """Create a GitHub issue."""
        try:
            if labels is None:
                labels = []
            
            url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues"
            
            data = {
                "title": title,
                "body": description,
                "labels": labels
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=data) as response:
                    if response.status == 201:
                        result = await response.json()
                        logger.info(f"Created GitHub issue: {title}")
                        return result["html_url"]
                    else:
                        logger.error(f"Failed to create GitHub issue: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error creating GitHub issue: {str(e)}")
            return None
    
    async def create_pull_request(self, title: str, description: str, head_branch: str, base_branch: str = "main") -> Optional[str]:
        """Create a GitHub pull request."""
        try:
            url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/pulls"
            
            data = {
                "title": title,
                "body": description,
                "head": head_branch,
                "base": base_branch
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=data) as response:
                    if response.status == 201:
                        result = await response.json()
                        logger.info(f"Created GitHub PR: {title}")
                        return result["html_url"]
                    else:
                        logger.error(f"Failed to create GitHub PR: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error creating GitHub PR: {str(e)}")
            return None
    
    async def update_issue_status(self, issue_number: int, status: str, comment: str = None) -> bool:
        """Update GitHub issue status."""
        try:
            # Add comment if provided
            if comment:
                await self.add_issue_comment(issue_number, comment)
            
            # Close issue if status is completed
            if status.upper() in ["COMPLETED", "CLOSED"]:
                url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues/{issue_number}"
                data = {"state": "closed"}
                
                async with aiohttp.ClientSession() as session:
                    async with session.patch(url, headers=self.headers, json=data) as response:
                        return response.status == 200
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating GitHub issue status: {str(e)}")
            return False
    
    async def add_issue_comment(self, issue_number: int, comment: str) -> bool:
        """Add a comment to a GitHub issue."""
        try:
            url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues/{issue_number}/comments"
            data = {"body": comment}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=data) as response:
                    return response.status == 201
                    
        except Exception as e:
            logger.error(f"Error adding GitHub comment: {str(e)}")
            return False

class SDLCBotTeam:
    """
    Autonomous SDLC Bot Team Management System.
    
    Features:
    - Self-organizing development teams
    - Automated GitHub integration
    - Cross-team communication protocols
    - Quality gates and approval workflows
    - Autonomous progress documentation
    """
    
    def __init__(self, ceo_portal: CEOPortal):
        """Initialize the SDLC Bot Team system."""
        self.ceo_portal = ceo_portal
        
        # Bot and team management
        self.bots: Dict[str, SDLCBot] = {}
        self.teams: Dict[str, SDLCTeam] = {}
        self.tasks: Dict[str, SDLCTask] = {}
        
        # GitHub integration
        self.github = None
        if hasattr(settings, 'github_token') and settings.github_token:
            self.github = GitHubIntegration(
                repo_owner=getattr(settings, 'github_owner', 'default'),
                repo_name=getattr(settings, 'github_repo', 'default'),
                token=settings.github_token
            )
        
        # Configuration
        self.max_tasks_per_bot = 3
        self.standup_interval = timedelta(days=1)
        self.sprint_duration = timedelta(weeks=2)
        
        # Team hierarchies
        self.team_hierarchies = {
            "Architecture": {
                "lead": BotRole.CHIEF_ARCHITECT,
                "members": [BotRole.SOLUTION_ARCHITECT, BotRole.SECURITY_ARCHITECT, BotRole.DATA_ARCHITECT]
            },
            "Development": {
                "lead": BotRole.DEV_LEAD,
                "members": [BotRole.SENIOR_DEV, BotRole.FULL_STACK_DEV, BotRole.BACKEND_DEV, BotRole.FRONTEND_DEV]
            },
            "Quality": {
                "lead": BotRole.QA_DIRECTOR,
                "members": [BotRole.UI_QA, BotRole.UX_QA, BotRole.PERFORMANCE_QA, BotRole.SECURITY_QA]
            },
            "Data": {
                "lead": BotRole.DBA_LEAD,
                "members": [BotRole.SENIOR_DBA, BotRole.DATA_ENGINEER, BotRole.ANALYTICS_DBA]
            },
            "Management": {
                "lead": BotRole.TECHNICAL_PM,
                "members": [BotRole.SCRUM_MASTER, BotRole.PRODUCT_OWNER, BotRole.BUSINESS_ANALYST, BotRole.TECHNICAL_WRITER]
            }
        }
        
        logger.info("SDLC Bot Team system initialized")
    
    async def initialize(self) -> bool:
        """Initialize the SDLC Bot Team system."""
        try:
            logger.info("Initializing SDLC Bot Team system...")
            
            # Create initial bot teams
            await self._create_initial_teams()
            
            # Start autonomous operations
            await self._start_autonomous_operations()
            
            logger.info("SDLC Bot Team system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing SDLC Bot Team: {str(e)}")
            return False
    
    async def create_bot(self, name: str, role: BotRole, team: str, specializations: List[str] = None) -> str:
        """Create a new SDLC bot."""
        try:
            if specializations is None:
                specializations = []
            
            bot_id = str(uuid.uuid4())
            
            # Default skills based on role
            skills = await self._get_default_skills_for_role(role)
            
            bot = SDLCBot(
                id=bot_id,
                name=name,
                role=role,
                team=team,
                specializations=specializations,
                skills=skills,
                current_tasks=[],
                completed_tasks=0,
                success_rate=0.8,  # Starting success rate
                availability=1.0,
                last_active=datetime.now(),
                status="ACTIVE",
                performance_metrics={},
                learning_progress={}
            )
            
            self.bots[bot_id] = bot
            
            # Add to team
            if team in self.teams:
                if role == self.team_hierarchies[team]["lead"]:
                    self.teams[team].lead_bot = bot_id
                else:
                    self.teams[team].member_bots.append(bot_id)
            
            logger.info(f"Created SDLC bot: {name} ({role.value}) in team {team}")
            return bot_id
            
        except Exception as e:
            logger.error(f"Error creating SDLC bot: {str(e)}")
            return ""
    
    async def assign_task(self, title: str, description: str, task_type: str, priority: TaskPriority, 
                         estimated_hours: int = 8, due_date: datetime = None) -> str:
        """Assign a task to the most suitable bot."""
        try:
            task_id = str(uuid.uuid4())
            
            if due_date is None:
                due_date = datetime.now() + timedelta(days=7)
            
            # Find best bot for the task
            assigned_bot, assigned_team = await self._find_best_bot_for_task(task_type, priority)
            
            task = SDLCTask(
                id=task_id,
                title=title,
                description=description,
                task_type=task_type,
                priority=priority,
                status=TaskStatus.PLANNED,
                assigned_bot=assigned_bot,
                assigned_team=assigned_team,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                due_date=due_date,
                estimated_hours=estimated_hours,
                actual_hours=0,
                dependencies=[],
                blocked_by=[],
                github_issue_url=None,
                github_pr_url=None,
                acceptance_criteria=[],
                progress_percentage=0,
                context={}
            )
            
            self.tasks[task_id] = task
            
            # Add to bot's current tasks
            if assigned_bot in self.bots:
                self.bots[assigned_bot].current_tasks.append(task_id)
            
            # Create GitHub issue if integration is available
            if self.github:
                github_url = await self.github.create_issue(
                    title=title,
                    description=f"{description}\n\n**Assigned to:** {assigned_bot}\n**Priority:** {priority.value}",
                    labels=[task_type.lower(), priority.value.lower()]
                )
                task.github_issue_url = github_url
            
            logger.info(f"Assigned task: {title} to bot {assigned_bot}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error assigning task: {str(e)}")
            return ""
    
    async def update_task_progress(self, task_id: str, progress_percentage: int, status: TaskStatus = None, 
                                  comments: str = None) -> bool:
        """Update task progress."""
        try:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            # Update progress
            task.progress_percentage = min(100, max(0, progress_percentage))
            task.updated_at = datetime.now()
            
            if status:
                task.status = status
            
            # Auto-complete if 100% progress
            if progress_percentage >= 100 and task.status != TaskStatus.COMPLETED:
                task.status = TaskStatus.COMPLETED
                
                # Remove from bot's current tasks
                if task.assigned_bot in self.bots:
                    bot = self.bots[task.assigned_bot]
                    if task_id in bot.current_tasks:
                        bot.current_tasks.remove(task_id)
                    bot.completed_tasks += 1
            
            # Update GitHub issue
            if self.github and task.github_issue_url and comments:
                # Extract issue number from URL
                issue_number = int(task.github_issue_url.split('/')[-1])
                await self.github.add_issue_comment(
                    issue_number,
                    f"**Progress Update:** {progress_percentage}%\n\n{comments}"
                )
                
                if task.status == TaskStatus.COMPLETED:
                    await self.github.update_issue_status(issue_number, "COMPLETED", "Task completed successfully!")
            
            logger.info(f"Updated task progress: {task.title} - {progress_percentage}%")
            return True
            
        except Exception as e:
            logger.error(f"Error updating task progress: {str(e)}")
            return False
    
    async def request_ceo_approval(self, title: str, description: str, financial_impact: float = 0,
                                  risk_level: float = 0, strategic_alignment: float = 0,
                                  requesting_bot: str = "SDLC_SYSTEM") -> Tuple[bool, str]:
        """Request CEO approval for a decision."""
        try:
            approved, message, priority = await self.ceo_portal.submit_decision_for_approval(
                requesting_bot=requesting_bot,
                title=title,
                description=description,
                category=DecisionCategory.TECHNOLOGY,
                financial_impact=financial_impact,
                risk_level=risk_level,
                strategic_alignment=strategic_alignment,
                context={"team": "SDLC", "system": "sdlc_bot_team"}
            )
            
            logger.info(f"CEO approval request: {title} - {message}")
            return approved, message
            
        except Exception as e:
            logger.error(f"Error requesting CEO approval: {str(e)}")
            return False, f"Error requesting approval: {str(e)}"
    
    async def conduct_daily_standup(self, team_name: str) -> Dict[str, Any]:
        """Conduct automated daily standup for a team."""
        try:
            if team_name not in self.teams:
                return {"error": "Team not found"}
            
            team = self.teams[team_name]
            standup_summary = {
                "team": team_name,
                "date": datetime.now().isoformat(),
                "participants": [],
                "completed_yesterday": [],
                "planned_today": [],
                "blockers": []
            }
            
            # Get updates from all team members
            for bot_id in [team.lead_bot] + team.member_bots:
                if bot_id in self.bots:
                    bot = self.bots[bot_id]
                    
                    # What was completed yesterday
                    completed_tasks = [
                        task for task in self.tasks.values()
                        if task.assigned_bot == bot_id and 
                        task.status == TaskStatus.COMPLETED and
                        task.updated_at > datetime.now() - timedelta(days=1)
                    ]
                    
                    # What's planned for today
                    current_tasks = [
                        task for task in self.tasks.values()
                        if task.assigned_bot == bot_id and 
                        task.status in [TaskStatus.IN_PROGRESS, TaskStatus.PLANNED]
                    ]
                    
                    # Blockers
                    blocked_tasks = [
                        task for task in self.tasks.values()
                        if task.assigned_bot == bot_id and 
                        task.status == TaskStatus.BLOCKED
                    ]
                    
                    standup_summary["participants"].append({
                        "bot": bot.name,
                        "role": bot.role.value,
                        "completed": [task.title for task in completed_tasks],
                        "planned": [task.title for task in current_tasks[:3]],  # Top 3
                        "blockers": [task.title for task in blocked_tasks]
                    })
                    
                    standup_summary["completed_yesterday"].extend([task.title for task in completed_tasks])
                    standup_summary["planned_today"].extend([task.title for task in current_tasks])
                    standup_summary["blockers"].extend([task.title for task in blocked_tasks])
            
            # Update team
            team.last_standup = datetime.now()
            
            # Create GitHub documentation
            if self.github:
                standup_doc = await self._create_standup_documentation(standup_summary)
                # Could create an issue or update wiki with standup notes
            
            logger.info(f"Conducted daily standup for team {team_name}")
            return standup_summary
            
        except Exception as e:
            logger.error(f"Error conducting standup: {str(e)}")
            return {"error": str(e)}
    
    async def get_team_status(self, team_name: str) -> Dict[str, Any]:
        """Get comprehensive team status."""
        try:
            if team_name not in self.teams:
                return {"error": "Team not found"}
            
            team = self.teams[team_name]
            
            # Collect team metrics
            active_bots = len([bot_id for bot_id in [team.lead_bot] + team.member_bots if bot_id in self.bots])
            
            # Task metrics
            team_tasks = [task for task in self.tasks.values() if task.assigned_team == team_name]
            total_tasks = len(team_tasks)
            completed_tasks = len([task for task in team_tasks if task.status == TaskStatus.COMPLETED])
            blocked_tasks = len([task for task in team_tasks if task.status == TaskStatus.BLOCKED])
            in_progress_tasks = len([task for task in team_tasks if task.status == TaskStatus.IN_PROGRESS])
            
            # Calculate health score
            completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
            blocker_rate = blocked_tasks / total_tasks if total_tasks > 0 else 0
            health_score = min(1.0, completion_rate * 0.7 + (1 - blocker_rate) * 0.3)
            
            # Recent achievements
            recent_achievements = [
                task.title for task in team_tasks
                if task.status == TaskStatus.COMPLETED and 
                task.updated_at > datetime.now() - timedelta(days=7)
            ]
            
            status = {
                "team_name": team_name,
                "team_type": "SDLC",
                "active_bots": active_bots,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "blocked_tasks": blocked_tasks,
                "in_progress_tasks": in_progress_tasks,
                "health_score": health_score,
                "last_update": datetime.now(),
                "key_metrics": {
                    "completion_rate": completion_rate,
                    "blocker_rate": blocker_rate,
                    "avg_task_time": await self._calculate_avg_task_time(team_name),
                    "success_rate": await self._calculate_team_success_rate(team_name)
                },
                "recent_achievements": recent_achievements[:5],  # Top 5
                "pending_ceo_items": 0  # Would be calculated from CEO portal
            }
            
            # Update CEO portal with team status
            await self.ceo_portal.update_team_status(f"SDLC_{team_name}", status)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting team status: {str(e)}")
            return {"error": str(e)}
    
    async def _create_initial_teams(self):
        """Create initial SDLC teams and bots."""
        try:
            for team_name, hierarchy in self.team_hierarchies.items():
                # Create team
                team = SDLCTeam(
                    name=team_name,
                    team_type="SDLC",
                    lead_bot="",
                    member_bots=[],
                    current_sprint=None,
                    active_tasks=[],
                    completed_tasks=[],
                    team_metrics={},
                    objectives=[],
                    last_standup=None
                )
                
                self.teams[team_name] = team
                
                # Create lead bot
                lead_bot_id = await self.create_bot(
                    name=f"{hierarchy['lead'].value}_001",
                    role=hierarchy["lead"],
                    team=team_name,
                    specializations=await self._get_specializations_for_role(hierarchy["lead"])
                )
                
                # Create member bots
                for i, member_role in enumerate(hierarchy["members"]):
                    await self.create_bot(
                        name=f"{member_role.value}_{i+1:03d}",
                        role=member_role,
                        team=team_name,
                        specializations=await self._get_specializations_for_role(member_role)
                    )
            
            logger.info("Created initial SDLC teams and bots")
            
        except Exception as e:
            logger.error(f"Error creating initial teams: {str(e)}")
    
    async def _start_autonomous_operations(self):
        """Start autonomous operations for SDLC teams."""
        try:
            # Start background tasks for autonomous operations
            asyncio.create_task(self._autonomous_task_management())
            asyncio.create_task(self._autonomous_standup_conductor())
            asyncio.create_task(self._autonomous_progress_tracking())
            
            logger.info("Started autonomous SDLC operations")
            
        except Exception as e:
            logger.error(f"Error starting autonomous operations: {str(e)}")
    
    async def _autonomous_task_management(self):
        """Autonomous task management and assignment."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Auto-assign unassigned tasks
                unassigned_tasks = [task for task in self.tasks.values() if not task.assigned_bot]
                
                for task in unassigned_tasks:
                    best_bot, best_team = await self._find_best_bot_for_task(task.task_type, task.priority)
                    if best_bot:
                        task.assigned_bot = best_bot
                        task.assigned_team = best_team
                        self.bots[best_bot].current_tasks.append(task.id)
                
                # Check for overdue tasks
                overdue_tasks = [
                    task for task in self.tasks.values()
                    if task.due_date and task.due_date < datetime.now() and task.status != TaskStatus.COMPLETED
                ]
                
                if overdue_tasks:
                    # Request CEO attention for critical overdue tasks
                    critical_overdue = [task for task in overdue_tasks if task.priority == TaskPriority.CRITICAL]
                    if critical_overdue:
                        await self.request_ceo_approval(
                            title=f"{len(critical_overdue)} Critical Tasks Overdue",
                            description=f"Critical tasks are overdue and may need intervention:\n" + 
                                      "\n".join([f"- {task.title} (due: {task.due_date})" for task in critical_overdue[:5]]),
                            risk_level=0.8,
                            strategic_alignment=0.7,
                            requesting_bot="AUTONOMOUS_TASK_MANAGER"
                        )
                
            except Exception as e:
                logger.error(f"Error in autonomous task management: {str(e)}")
    
    async def _autonomous_standup_conductor(self):
        """Conduct autonomous daily standups."""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                # Conduct standups for all teams
                for team_name in self.teams.keys():
                    await self.conduct_daily_standup(team_name)
                
            except Exception as e:
                logger.error(f"Error in autonomous standup conductor: {str(e)}")
    
    async def _autonomous_progress_tracking(self):
        """Autonomous progress tracking and updates."""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                # Update team statuses
                for team_name in self.teams.keys():
                    await self.get_team_status(team_name)
                
                # Simulate bot progress (in real implementation, bots would report their own progress)
                for task in self.tasks.values():
                    if task.status == TaskStatus.IN_PROGRESS:
                        # Simulate progress increment
                        progress_increment = 5  # 5% per 30 minutes for active tasks
                        new_progress = min(100, task.progress_percentage + progress_increment)
                        await self.update_task_progress(task.id, new_progress)
                
            except Exception as e:
                logger.error(f"Error in autonomous progress tracking: {str(e)}")
    
    async def _find_best_bot_for_task(self, task_type: str, priority: TaskPriority) -> Tuple[str, str]:
        """Find the best bot for a specific task."""
        try:
            # Map task types to preferred roles
            task_role_mapping = {
                "FEATURE": [BotRole.FULL_STACK_DEV, BotRole.SENIOR_DEV, BotRole.BACKEND_DEV],
                "BUG": [BotRole.SENIOR_DEV, BotRole.FULL_STACK_DEV],
                "TECH_DEBT": [BotRole.DEV_LEAD, BotRole.SENIOR_DEV],
                "RESEARCH": [BotRole.SOLUTION_ARCHITECT, BotRole.SENIOR_DEV],
                "DOCUMENTATION": [BotRole.TECHNICAL_WRITER, BotRole.DEV_LEAD],
                "ARCHITECTURE": [BotRole.CHIEF_ARCHITECT, BotRole.SOLUTION_ARCHITECT],
                "SECURITY": [BotRole.SECURITY_ARCHITECT, BotRole.SECURITY_QA],
                "DATABASE": [BotRole.DBA_LEAD, BotRole.SENIOR_DBA],
                "TESTING": [BotRole.QA_DIRECTOR, BotRole.UI_QA, BotRole.UX_QA]
            }
            
            preferred_roles = task_role_mapping.get(task_type, [BotRole.FULL_STACK_DEV])
            
            # Find available bots with preferred roles
            available_bots = []
            
            for bot_id, bot in self.bots.items():
                if (bot.role in preferred_roles and 
                    bot.status == "ACTIVE" and 
                    len(bot.current_tasks) < self.max_tasks_per_bot and
                    bot.availability > 0.5):
                    
                    # Calculate suitability score
                    score = bot.success_rate * 0.4 + bot.availability * 0.3
                    
                    # Bonus for relevant skills
                    relevant_skills = [skill for skill in bot.skills.keys() if task_type.lower() in skill.lower()]
                    if relevant_skills:
                        score += sum(bot.skills[skill] for skill in relevant_skills) / len(relevant_skills) * 0.3
                    
                    available_bots.append((bot_id, bot.team, score))
            
            if available_bots:
                # Sort by score and return best bot
                available_bots.sort(key=lambda x: x[2], reverse=True)
                return available_bots[0][0], available_bots[0][1]
            
            # Fallback: return any available bot
            for bot_id, bot in self.bots.items():
                if bot.status == "ACTIVE" and len(bot.current_tasks) < self.max_tasks_per_bot:
                    return bot_id, bot.team
            
            return "", ""
            
        except Exception as e:
            logger.error(f"Error finding best bot for task: {str(e)}")
            return "", ""
    
    async def _get_default_skills_for_role(self, role: BotRole) -> Dict[str, float]:
        """Get default skills for a bot role."""
        skill_mappings = {
            # Architecture roles
            BotRole.CHIEF_ARCHITECT: {"system_design": 0.9, "technology_strategy": 0.9, "leadership": 0.8},
            BotRole.SOLUTION_ARCHITECT: {"solution_design": 0.9, "integration": 0.8, "documentation": 0.7},
            BotRole.SECURITY_ARCHITECT: {"security_design": 0.9, "risk_assessment": 0.8, "compliance": 0.8},
            BotRole.DATA_ARCHITECT: {"data_modeling": 0.9, "database_design": 0.9, "analytics": 0.7},
            
            # Development roles
            BotRole.DEV_LEAD: {"leadership": 0.8, "code_review": 0.9, "mentoring": 0.8, "architecture": 0.7},
            BotRole.SENIOR_DEV: {"programming": 0.9, "debugging": 0.9, "optimization": 0.8, "mentoring": 0.6},
            BotRole.FULL_STACK_DEV: {"frontend": 0.8, "backend": 0.8, "database": 0.7, "integration": 0.7},
            BotRole.BACKEND_DEV: {"backend": 0.9, "api_design": 0.8, "database": 0.8, "performance": 0.7},
            BotRole.FRONTEND_DEV: {"frontend": 0.9, "ui_ux": 0.8, "javascript": 0.9, "responsive_design": 0.8},
            
            # Quality roles
            BotRole.QA_DIRECTOR: {"test_strategy": 0.9, "quality_management": 0.9, "leadership": 0.8},
            BotRole.UI_QA: {"ui_testing": 0.9, "automation": 0.8, "usability": 0.7},
            BotRole.UX_QA: {"ux_testing": 0.9, "user_research": 0.8, "accessibility": 0.8},
            BotRole.PERFORMANCE_QA: {"performance_testing": 0.9, "load_testing": 0.8, "optimization": 0.7},
            BotRole.SECURITY_QA: {"security_testing": 0.9, "vulnerability_assessment": 0.8, "penetration_testing": 0.7},
            
            # Data roles
            BotRole.DBA_LEAD: {"database_administration": 0.9, "performance_tuning": 0.8, "leadership": 0.7},
            BotRole.SENIOR_DBA: {"database_administration": 0.9, "backup_recovery": 0.8, "monitoring": 0.8},
            BotRole.DATA_ENGINEER: {"etl_development": 0.9, "data_pipelines": 0.8, "big_data": 0.7},
            BotRole.ANALYTICS_DBA: {"analytics": 0.9, "reporting": 0.8, "data_visualization": 0.7},
            
            # Management roles
            BotRole.TECHNICAL_PM: {"project_management": 0.9, "coordination": 0.8, "communication": 0.9},
            BotRole.SCRUM_MASTER: {"scrum": 0.9, "facilitation": 0.8, "coaching": 0.8},
            BotRole.PRODUCT_OWNER: {"product_management": 0.9, "requirements": 0.8, "stakeholder_management": 0.8},
            BotRole.BUSINESS_ANALYST: {"business_analysis": 0.9, "requirements_gathering": 0.8, "documentation": 0.8},
            BotRole.TECHNICAL_WRITER: {"technical_writing": 0.9, "documentation": 0.9, "communication": 0.8}
        }
        
        return skill_mappings.get(role, {"general": 0.7})
    
    async def _get_specializations_for_role(self, role: BotRole) -> List[str]:
        """Get specializations for a bot role."""
        specialization_mappings = {
            BotRole.CHIEF_ARCHITECT: ["Enterprise Architecture", "Cloud Architecture", "Microservices"],
            BotRole.SOLUTION_ARCHITECT: ["System Integration", "API Design", "Cloud Solutions"],
            BotRole.SECURITY_ARCHITECT: ["Zero Trust", "DevSecOps", "Compliance"],
            BotRole.DATA_ARCHITECT: ["Data Lakes", "Analytics", "ML Pipelines"],
            BotRole.DEV_LEAD: ["Team Leadership", "Code Quality", "Technical Strategy"],
            BotRole.SENIOR_DEV: ["Python", "JavaScript", "System Design"],
            BotRole.FULL_STACK_DEV: ["React", "Node.js", "PostgreSQL"],
            BotRole.BACKEND_DEV: ["API Development", "Database Design", "Performance"],
            BotRole.FRONTEND_DEV: ["React", "Vue.js", "CSS/SCSS"],
            BotRole.QA_DIRECTOR: ["Test Automation", "Quality Strategy", "CI/CD"],
            BotRole.UI_QA: ["Selenium", "Cypress", "Visual Testing"],
            BotRole.UX_QA: ["User Testing", "Accessibility", "Usability"],
            BotRole.PERFORMANCE_QA: ["JMeter", "Load Testing", "APM"],
            BotRole.SECURITY_QA: ["OWASP", "Penetration Testing", "Security Auditing"],
            BotRole.DBA_LEAD: ["PostgreSQL", "Performance Tuning", "High Availability"],
            BotRole.SENIOR_DBA: ["Database Optimization", "Backup Strategies", "Monitoring"],
            BotRole.DATA_ENGINEER: ["Apache Spark", "Kafka", "Data Pipelines"],
            BotRole.ANALYTICS_DBA: ["Data Warehousing", "BI Tools", "Analytics"],
            BotRole.TECHNICAL_PM: ["Agile", "Stakeholder Management", "Risk Management"],
            BotRole.SCRUM_MASTER: ["Scrum", "Kanban", "Team Facilitation"],
            BotRole.PRODUCT_OWNER: ["Product Strategy", "User Stories", "Roadmapping"],
            BotRole.BUSINESS_ANALYST: ["Requirements Engineering", "Process Analysis", "Stakeholder Communication"],
            BotRole.TECHNICAL_WRITER: ["API Documentation", "User Guides", "Technical Communication"]
        }
        
        return specialization_mappings.get(role, ["General"])
    
    async def _calculate_avg_task_time(self, team_name: str) -> float:
        """Calculate average task completion time for a team."""
        try:
            team_tasks = [
                task for task in self.tasks.values()
                if task.assigned_team == team_name and task.status == TaskStatus.COMPLETED
            ]
            
            if not team_tasks:
                return 0.0
            
            total_time = sum(task.actual_hours for task in team_tasks)
            return total_time / len(team_tasks)
            
        except Exception as e:
            logger.error(f"Error calculating average task time: {str(e)}")
            return 0.0
    
    async def _calculate_team_success_rate(self, team_name: str) -> float:
        """Calculate success rate for a team."""
        try:
            team_tasks = [task for task in self.tasks.values() if task.assigned_team == team_name]
            
            if not team_tasks:
                return 0.5
            
            completed_tasks = [task for task in team_tasks if task.status == TaskStatus.COMPLETED]
            return len(completed_tasks) / len(team_tasks)
            
        except Exception as e:
            logger.error(f"Error calculating team success rate: {str(e)}")
            return 0.5
    
    async def _create_standup_documentation(self, standup_summary: Dict[str, Any]) -> str:
        """Create documentation for standup meeting."""
        try:
            doc = f"""# Daily Standup - {standup_summary['team']}
**Date:** {standup_summary['date']}

## Team Updates

"""
            
            for participant in standup_summary['participants']:
                doc += f"""### {participant['bot']} ({participant['role']})

**Completed Yesterday:**
{chr(10).join([f"- {item}" for item in participant['completed']])}

**Planned for Today:**
{chr(10).join([f"- {item}" for item in participant['planned']])}

**Blockers:**
{chr(10).join([f"- {item}" for item in participant['blockers']])}

"""
            
            if standup_summary['blockers']:
                doc += f"""## Action Items
{chr(10).join([f"- Resolve: {blocker}" for blocker in set(standup_summary['blockers'])])}
"""
            
            return doc
            
        except Exception as e:
            logger.error(f"Error creating standup documentation: {str(e)}")
            return "" 