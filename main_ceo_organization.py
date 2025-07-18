"""
CEO-Centric AI Organization Management System - Industry-Agnostic
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import settings
from core.ceo_portal import CEOPortal, DecisionCategory, Priority
from core.sdlc_bot_team import SDLCBotTeam, TaskPriority
from core.cross_team_coordinator import CrossTeamCoordinator
from core.ai_engine import AIEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for the organization
organization_state = {
    "ceo_portal": None,
    "sdlc_team": None,
    "business_engine": None,
    "coordinator": None,
    "business_domain": "Business Domain",
    "industry": "Software Development",
    "is_initialized": False
}

# Pydantic models for API
class InitializeRequest(BaseModel):
    business_domain: str = "Business Domain"
    industry: str = "Software Development"
    business_context: Optional[Dict[str, Any]] = None

class TaskRequest(BaseModel):
    title: str
    description: str
    task_type: str
    priority: str
    estimated_hours: int = 8

class CoordinationRequest(BaseModel):
    requesting_team: str
    target_team: str
    coordination_type: str
    title: str
    description: str
    strategic_value: float = 0.5

class BusinessContextUpdate(BaseModel):
    domain_name: str
    industry: str
    capabilities: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    strategic_goals: Optional[Dict[str, Any]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting CEO-Centric AI Organization System...")
    yield
    logger.info("Shutting down CEO-Centric AI Organization System...")

# FastAPI app
app = FastAPI(
    title="CEO-Centric AI Organization API",
    description="Industry-Agnostic CEO-operated AI organization with autonomous SDLC and business domain teams",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/initialize")
async def initialize_organization(request: InitializeRequest):
    """Initialize the CEO-centric AI organization for any industry."""
    try:
        global organization_state
        
        logger.info(f"Initializing organization for {request.industry} industry, domain: {request.business_domain}")
        
        # Initialize CEO Portal
        ceo_portal = CEOPortal()
        await ceo_portal.initialize()
        
        # Initialize Business Engine (industry-agnostic AI engine)
        business_engine = AIEngine()
        await business_engine.initialize()
        
        # Initialize SDLC Bot Team with project context
        project_context = {
            "industry": request.industry,
            "business_domain": request.business_domain,
            "project_type": f"{request.industry} Application Development"
        }
        if request.business_context:
            project_context.update(request.business_context)
        
        sdlc_team = SDLCBotTeam(ceo_portal, project_context)
        await sdlc_team.initialize()
        
        # Initialize Cross-Team Coordinator
        coordinator = CrossTeamCoordinator(
            ceo_portal, 
            sdlc_team, 
            business_engine, 
            request.business_domain
        )
        
        # Set business domain context if provided
        if request.business_context:
            coordinator.set_business_domain(request.business_domain, request.business_context)
        
        await coordinator.initialize()
        
        # Update global state
        organization_state.update({
            "ceo_portal": ceo_portal,
            "sdlc_team": sdlc_team,
            "business_engine": business_engine,
            "coordinator": coordinator,
            "business_domain": request.business_domain,
            "industry": request.industry,
            "is_initialized": True
        })
        
        logger.info(f"Organization initialized successfully for {request.industry}")
        
        return {
            "status": "success",
            "message": f"CEO-centric AI organization initialized for {request.industry}",
            "business_domain": request.business_domain,
            "industry": request.industry,
            "teams": {
                "ceo_portal": "active",
                "sdlc_team": "active",
                "business_engine": "active",
                "coordinator": "active"
            }
        }
        
    except Exception as e:
        logger.error(f"Error initializing organization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.post("/api/v1/update-business-context")
async def update_business_context(request: BusinessContextUpdate):
    """Update business context for industry customization."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        # Update business domain in coordinator
        coordinator = organization_state["coordinator"]
        if coordinator:
            business_context = {}
            if request.capabilities:
                business_context["capabilities"] = request.capabilities
            if request.performance_metrics:
                business_context["performance_metrics"] = request.performance_metrics
            if request.strategic_goals:
                business_context["strategic_goals"] = request.strategic_goals
            
            coordinator.set_business_domain(request.domain_name, business_context)
        
        # Update SDLC team project context
        sdlc_team = organization_state["sdlc_team"]
        if sdlc_team:
            project_context = {
                "industry": request.industry,
                "business_domain": request.domain_name,
                "project_type": f"{request.industry} Application Development"
            }
            if request.capabilities:
                project_context["business_capabilities"] = request.capabilities
            
            sdlc_team.set_project_context(project_context)
        
        # Update global state
        organization_state["business_domain"] = request.domain_name
        organization_state["industry"] = request.industry
        
        logger.info(f"Business context updated for {request.industry} - {request.domain_name}")
        
        return {
            "status": "success",
            "message": f"Business context updated for {request.industry}",
            "business_domain": request.domain_name,
            "industry": request.industry
        }
        
    except Exception as e:
        logger.error(f"Error updating business context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Context update failed: {str(e)}")

@app.get("/api/v1/status")
async def get_organization_status():
    """Get comprehensive organization status."""
    try:
        if not organization_state["is_initialized"]:
            return {
                "status": "not_initialized",
                "message": "Organization not initialized"
            }
        
        ceo_portal = organization_state["ceo_portal"]
        sdlc_team = organization_state["sdlc_team"]
        coordinator = organization_state["coordinator"]
        
        # Get team statuses
        sdlc_statuses = {}
        for team_name in ["Architecture", "Development", "Quality", "Data", "Management"]:
            sdlc_statuses[team_name] = await sdlc_team.get_team_status(team_name)
        
        # Get coordination status
        coordination_status = await coordinator.monitor_cross_team_performance()
        
        # Get CEO portal status
        ceo_status = await ceo_portal.get_dashboard_data()
        
        return {
            "status": "active",
            "business_domain": organization_state["business_domain"],
            "industry": organization_state["industry"],
            "ceo_portal": ceo_status,
            "sdlc_teams": sdlc_statuses,
            "coordination": coordination_status,
            "last_update": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting organization status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@app.post("/api/v1/tasks/assign")
async def assign_task(request: TaskRequest):
    """Assign a task to the SDLC team."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        sdlc_team = organization_state["sdlc_team"]
        
        # Convert priority string to enum
        priority_map = {
            "CRITICAL": TaskPriority.CRITICAL,
            "HIGH": TaskPriority.HIGH,
            "MEDIUM": TaskPriority.MEDIUM,
            "LOW": TaskPriority.LOW
        }
        
        priority = priority_map.get(request.priority.upper(), TaskPriority.MEDIUM)
        
        # Assign task with industry context
        task_id = await sdlc_team.assign_task(
            title=request.title,
            description=request.description,
            task_type=request.task_type,
            priority=priority,
            estimated_hours=request.estimated_hours,
            industry_context={
                "industry": organization_state["industry"],
                "business_domain": organization_state["business_domain"]
            }
        )
        
        if task_id:
            logger.info(f"Task assigned: {request.title} -> {task_id}")
            return {
                "status": "success",
                "task_id": task_id,
                "message": f"Task '{request.title}' assigned successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Task assignment failed")
        
    except Exception as e:
        logger.error(f"Error assigning task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task assignment failed: {str(e)}")

@app.post("/api/v1/coordination/request")
async def request_coordination(request: CoordinationRequest):
    """Request cross-team coordination."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        coordinator = organization_state["coordinator"]
        
        # Convert coordination type string to enum
        from core.cross_team_coordinator import CoordinationType
        coordination_type_map = {
            "KNOWLEDGE_TRANSFER": CoordinationType.KNOWLEDGE_TRANSFER,
            "RESOURCE_SHARING": CoordinationType.RESOURCE_SHARING,
            "STRATEGIC_ALIGNMENT": CoordinationType.STRATEGIC_ALIGNMENT,
            "TECHNOLOGY_SYNC": CoordinationType.TECHNOLOGY_SYNC,
            "DATA_INTEGRATION": CoordinationType.DATA_INTEGRATION,
            "PERFORMANCE_OPTIMIZATION": CoordinationType.PERFORMANCE_OPTIMIZATION
        }
        
        coordination_type = coordination_type_map.get(
            request.coordination_type.upper(), 
            CoordinationType.KNOWLEDGE_TRANSFER
        )
        
        # Request coordination
        approved, message = await coordinator.request_coordination(
            requesting_team=request.requesting_team,
            target_team=request.target_team,
            coordination_type=coordination_type,
            title=request.title,
            description=request.description,
            strategic_value=request.strategic_value
        )
        
        logger.info(f"Coordination request: {request.title} - {message}")
        
        return {
            "status": "success" if approved else "pending",
            "approved": approved,
            "message": message,
            "coordination_type": request.coordination_type,
            "business_domain": organization_state["business_domain"]
        }
        
    except Exception as e:
        logger.error(f"Error requesting coordination: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Coordination request failed: {str(e)}")

@app.get("/api/v1/ceo/decisions")
async def get_ceo_decisions():
    """Get pending CEO decisions."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        ceo_portal = organization_state["ceo_portal"]
        decisions = await ceo_portal.get_pending_decisions()
        
        return {
            "status": "success",
            "pending_decisions": decisions,
            "business_domain": organization_state["business_domain"],
            "industry": organization_state["industry"]
        }
        
    except Exception as e:
        logger.error(f"Error getting CEO decisions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get decisions: {str(e)}")

@app.post("/api/v1/ceo/decisions/{decision_id}/approve")
async def approve_decision(decision_id: str):
    """Approve a CEO decision."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        ceo_portal = organization_state["ceo_portal"]
        success = await ceo_portal.approve_decision(decision_id)
        
        if success:
            logger.info(f"Decision approved: {decision_id}")
            return {
                "status": "success",
                "message": f"Decision {decision_id} approved",
                "decision_id": decision_id
            }
        else:
            raise HTTPException(status_code=404, detail="Decision not found or already processed")
        
    except Exception as e:
        logger.error(f"Error approving decision: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Decision approval failed: {str(e)}")

@app.post("/api/v1/ceo/decisions/{decision_id}/reject")
async def reject_decision(decision_id: str):
    """Reject a CEO decision."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        ceo_portal = organization_state["ceo_portal"]
        success = await ceo_portal.reject_decision(decision_id)
        
        if success:
            logger.info(f"Decision rejected: {decision_id}")
            return {
                "status": "success",
                "message": f"Decision {decision_id} rejected",
                "decision_id": decision_id
            }
        else:
            raise HTTPException(status_code=404, detail="Decision not found or already processed")
        
    except Exception as e:
        logger.error(f"Error rejecting decision: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Decision rejection failed: {str(e)}")

@app.get("/api/v1/synergies")
async def get_synergy_opportunities():
    """Get identified synergy opportunities."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        coordinator = organization_state["coordinator"]
        opportunities = await coordinator.identify_synergy_opportunities()
        
        return {
            "status": "success",
            "synergy_opportunities": [
                {
                    "id": op.id,
                    "title": op.title,
                    "description": op.description,
                    "teams_involved": op.teams_involved,
                    "potential_value": op.potential_value,
                    "success_probability": op.success_probability,
                    "opportunity_type": op.opportunity_type,
                    "status": op.status
                }
                for op in opportunities
            ],
            "business_domain": organization_state["business_domain"],
            "total_opportunities": len(opportunities)
        }
        
    except Exception as e:
        logger.error(f"Error getting synergy opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get synergies: {str(e)}")

@app.post("/api/v1/command")
async def process_command(command: Dict[str, Any]):
    """Process natural language command for the organization."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        command_text = command.get("text", "").lower()
        
        # Simple command processing (can be enhanced with NLP)
        if "status" in command_text:
            return await get_organization_status()
        elif "synergies" in command_text or "opportunities" in command_text:
            return await get_synergy_opportunities()
        elif "decisions" in command_text:
            return await get_ceo_decisions()
        elif "assign task" in command_text:
            # Extract task details from command (simplified)
            return {
                "status": "info",
                "message": "Please use the /api/v1/tasks/assign endpoint with task details",
                "business_domain": organization_state["business_domain"]
            }
        else:
            return {
                "status": "info",
                "message": f"Command received for {organization_state['business_domain']} organization. Available commands: status, synergies, decisions, assign task",
                "business_domain": organization_state["business_domain"],
                "industry": organization_state["industry"]
            }
        
    except Exception as e:
        logger.error(f"Error processing command: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Command processing failed: {str(e)}")

# Add new API endpoints after the existing ones

@app.get("/api/v1/bots/status")
async def get_bot_status():
    """Get real-time status of all bots with detailed monitoring data."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        ceo_portal = organization_state["ceo_portal"]
        
        # Get bot monitoring dashboard
        bot_dashboard = await ceo_portal.get_bot_monitoring_dashboard()
        
        return {
            "status": "success",
            "bot_monitoring": bot_dashboard,
            "business_domain": organization_state["business_domain"],
            "industry": organization_state["industry"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting bot status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get bot status: {str(e)}")

@app.post("/api/v1/bots/{bot_id}/update")
async def update_bot_status(bot_id: str, status_update: Dict[str, Any]):
    """Update bot status in real-time."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        ceo_portal = organization_state["ceo_portal"]
        
        # Update bot status
        success = await ceo_portal.update_bot_status(bot_id, status_update)
        
        if success:
            return {
                "status": "success",
                "message": f"Bot {bot_id} status updated",
                "bot_id": bot_id
            }
        else:
            raise HTTPException(status_code=404, detail="Bot not found")
        
    except Exception as e:
        logger.error(f"Error updating bot status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update bot status: {str(e)}")

@app.post("/api/v1/self-improvement/request")
async def submit_self_improvement_request(request: Dict[str, Any]):
    """Submit a self-improvement request for Bot Builder."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        # Validate required fields
        required_fields = ["title", "description", "component", "priority"]
        for field in required_fields:
            if field not in request:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        ceo_portal = organization_state["ceo_portal"]
        
        # Import SystemComponent and Priority enums
        from core.ceo_portal import SystemComponent, Priority
        
        # Convert component and priority strings to enums
        try:
            component = SystemComponent[request["component"].upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid component: {request['component']}")
        
        try:
            priority = Priority[request["priority"].upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {request['priority']}")
        
        # Submit self-improvement request
        success, message = await ceo_portal.submit_self_improvement_request(
            title=request["title"],
            description=request["description"],
            component=component,
            priority=priority,
            requesting_entity=request.get("requesting_entity", "CEO"),
            expected_benefits=request.get("expected_benefits", []),
            success_criteria=request.get("success_criteria", [])
        )
        
        return {
            "status": "success" if success else "pending",
            "approved": success,
            "message": message,
            "improvement_request": {
                "title": request["title"],
                "component": request["component"],
                "priority": request["priority"]
            },
            "business_domain": organization_state["business_domain"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting self-improvement request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit request: {str(e)}")

@app.get("/api/v1/self-improvement/status")
async def get_self_improvement_status():
    """Get status of all self-improvement requests."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        ceo_portal = organization_state["ceo_portal"]
        
        # Get self-improvement status
        improvement_status = await ceo_portal.get_self_improvement_status()
        
        return {
            "status": "success",
            "self_improvement": improvement_status,
            "business_domain": organization_state["business_domain"],
            "industry": organization_state["industry"]
        }
        
    except Exception as e:
        logger.error(f"Error getting self-improvement status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get improvement status: {str(e)}")

@app.get("/api/v1/dashboard/enhanced")
async def get_enhanced_dashboard():
    """Get enhanced CEO dashboard with bot monitoring and self-improvement data."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        ceo_portal = organization_state["ceo_portal"]
        
        # Get enhanced dashboard data
        dashboard_data = await ceo_portal.get_enhanced_dashboard_data()
        
        return {
            "status": "success",
            "dashboard": dashboard_data,
            "business_domain": organization_state["business_domain"],
            "industry": organization_state["industry"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard: {str(e)}")

@app.post("/api/v1/bots/register")
async def register_bot_monitoring(bot_data: Dict[str, Any]):
    """Register a bot for monitoring."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        ceo_portal = organization_state["ceo_portal"]
        
        # Register bot for monitoring
        success = await ceo_portal.register_bot_monitoring(bot_data)
        
        if success:
            return {
                "status": "success",
                "message": f"Bot {bot_data.get('name', 'Unknown')} registered for monitoring",
                "bot_id": bot_data.get('id')
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to register bot")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering bot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to register bot: {str(e)}")

@app.get("/api/v1/bots/chart-data")
async def get_bot_chart_data():
    """Get bot data formatted for charts and visualizations."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        sdlc_team = organization_state["sdlc_team"]
        
        # Get bot data from SDLC team
        bot_chart_data = {}
        team_names = ["Architecture", "Development", "Quality", "Data", "Management"]
        
        for team_name in team_names:
            team_bots = []
            for bot_id, bot in sdlc_team.bots.items():
                if bot.team == team_name:
                    team_bots.append({
                        'id': bot.id,
                        'name': bot.name,
                        'role': bot.role.value,
                        'status': bot.status,
                        'availability': bot.availability,
                        'success_rate': bot.success_rate,
                        'current_tasks': bot.current_tasks,
                        'completed_tasks': bot.completed_tasks,
                        'specializations': bot.specializations,
                        'last_active': bot.last_active.isoformat()
                    })
            
            bot_chart_data[team_name] = {
                'bots': team_bots,
                'total_bots': len(team_bots),
                'active_bots': len([b for b in team_bots if b['status'] == 'ACTIVE']),
                'avg_availability': sum(b['availability'] for b in team_bots) / len(team_bots) if team_bots else 0
            }
        
        return {
            "status": "success",
            "chart_data": bot_chart_data,
            "summary": {
                "total_bots": sum(team['total_bots'] for team in bot_chart_data.values()),
                "total_active": sum(team['active_bots'] for team in bot_chart_data.values()),
                "overall_availability": sum(team['avg_availability'] for team in bot_chart_data.values()) / len(bot_chart_data) if bot_chart_data else 0
            },
            "business_domain": organization_state["business_domain"],
            "industry": organization_state["industry"]
        }
        
    except Exception as e:
        logger.error(f"Error getting bot chart data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chart data: {str(e)}")

@app.post("/api/v1/bot-builder/self-enhance")
async def bot_builder_self_enhance(enhancement_request: Dict[str, Any]):
    """Bot Builder requests self-enhancement using its own SDLC team."""
    try:
        if not organization_state["is_initialized"]:
            raise HTTPException(status_code=400, detail="Organization not initialized")
        
        # Validate request
        if "description" not in enhancement_request:
            raise HTTPException(status_code=400, detail="Enhancement description required")
        
        sdlc_team = organization_state["sdlc_team"]
        ceo_portal = organization_state["ceo_portal"]
        
        description = enhancement_request["description"]
        priority = enhancement_request.get("priority", "MEDIUM")
        component = enhancement_request.get("component", "CEO_PORTAL")
        
        # Submit to SDLC team as a self-improvement task
        from core.sdlc_bot_team import TaskPriority
        
        priority_map = {
            "CRITICAL": TaskPriority.CRITICAL,
            "HIGH": TaskPriority.HIGH,
            "MEDIUM": TaskPriority.MEDIUM,
            "LOW": TaskPriority.LOW
        }
        
        task_priority = priority_map.get(priority.upper(), TaskPriority.MEDIUM)
        
        # Create recursive self-improvement task
        task_id = await sdlc_team.assign_task(
            title=f"🧠 Bot Builder Self-Enhancement: {description[:50]}...",
            description=f"""
🔄 **RECURSIVE SELF-IMPROVEMENT REQUEST**

**Enhancement Description:** {description}

**Target Component:** {component}

**Self-Improvement Context:**
This is a meta-enhancement where Bot Builder is using its own SDLC team to improve itself based on strategic requirements and user feedback.

**Implementation Approach:**
1. Analyze current {component} implementation
2. Identify specific improvement opportunities  
3. Design enhanced solution architecture
4. Implement improvements with backward compatibility
5. Test thoroughly across all use cases
6. Document changes and update system

**Success Criteria:**
- Enhanced functionality delivered as requested
- No regression in existing capabilities
- Improved user experience and system performance
- Self-improvement capability demonstrated

**Recursive Enhancement Note:** 
This task demonstrates Bot Builder's ability to evolve itself through autonomous decision-making and implementation via its own development team.
            """,
            task_type="FEATURE",
            priority=task_priority,
            estimated_hours=enhancement_request.get("estimated_hours", 12),
            industry_context={
                "industry": "AI Systems Development", 
                "business_domain": "Bot Builder Self-Enhancement",
                "enhancement_type": "recursive_improvement",
                "target_component": component,
                "requesting_entity": "BOT_BUILDER_CORE"
            }
        )
        
        # Also submit to CEO portal as self-improvement request
        from core.ceo_portal import SystemComponent, Priority as CEOPriority
        
        try:
            system_component = SystemComponent[component.upper()]
        except KeyError:
            system_component = SystemComponent.CEO_PORTAL
        
        try:
            ceo_priority = CEOPriority[priority.upper()]
        except KeyError:
            ceo_priority = CEOPriority.MEDIUM
        
        ceo_success, ceo_message = await ceo_portal.submit_self_improvement_request(
            title=f"Bot Builder Self-Enhancement: {description[:50]}...",
            description=description,
            component=system_component,
            priority=ceo_priority,
            requesting_entity="BOT_BUILDER_CORE",
            expected_benefits=enhancement_request.get("expected_benefits", [
                "Improved system capabilities",
                "Enhanced user experience", 
                "Increased operational efficiency"
            ]),
            success_criteria=enhancement_request.get("success_criteria", [
                "Enhancement successfully implemented",
                "No performance degradation",
                "User requirements fulfilled"
            ])
        )
        
        return {
            "status": "success",
            "message": "🧠 Bot Builder self-enhancement initiated",
            "sdlc_task_id": task_id,
            "ceo_portal_submitted": ceo_success,
            "ceo_message": ceo_message,
            "enhancement_details": {
                "description": description,
                "component": component,
                "priority": priority,
                "estimated_hours": enhancement_request.get("estimated_hours", 12)
            },
            "meta_insight": "🔄 Bot Builder is now using its own SDLC team to recursively improve itself!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Bot Builder self-enhancement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Self-enhancement failed: {str(e)}")

# Update the root endpoint to include new features
@app.get("/")
async def root():
    """Root endpoint with organization information."""
    return {
        "message": "🤖 Bot Builder CEO Portal - Industry Agnostic with Self-Improvement",
        "version": "2.1.0",
        "business_domain": organization_state.get("business_domain", "Not Set"),
        "industry": organization_state.get("industry", "Not Set"),
        "status": "active" if organization_state["is_initialized"] else "not_initialized",
        "documentation": "/docs",
        "features": [
            "🤖 Industry-agnostic SDLC bot teams",
            "🏢 Configurable business domain integration", 
            "🤝 Cross-team coordination and synergy identification",
            "👔 CEO decision portal with intelligent filtering",
            "📋 Autonomous progress documentation",
            "📊 Strategic alignment monitoring",
            "📈 Real-time performance dashboards",
            "🎯 Interactive bot status visualization",
            "🧠 Recursive self-improvement capabilities",
            "📱 Enhanced CEO portal with bot monitoring",
            "🔄 Bot Builder can improve itself using its own SDLC team"
        ],
        "new_endpoints": [
            "GET /api/v1/bots/status - Real-time bot monitoring",
            "GET /api/v1/bots/chart-data - Bot visualization data",
            "POST /api/v1/self-improvement/request - Submit improvement requests",
            "GET /api/v1/self-improvement/status - Track improvement progress", 
            "POST /api/v1/bot-builder/self-enhance - Recursive self-improvement",
            "GET /api/v1/dashboard/enhanced - Enhanced CEO dashboard"
        ]
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "business_domain": organization_state.get("business_domain", "Not Set"),
            "industry": organization_state.get("industry", "Not Set"),
            "components": {
                "ceo_portal": "active" if organization_state.get("ceo_portal") else "inactive",
                "sdlc_team": "active" if organization_state.get("sdlc_team") else "inactive",
                "business_engine": "active" if organization_state.get("business_engine") else "inactive",
                "coordinator": "active" if organization_state.get("coordinator") else "inactive"
            },
            "initialized": organization_state["is_initialized"]
        }
        
        # Check component health
        if organization_state["is_initialized"]:
            # Could add more detailed health checks here
            health_status["overall_health"] = "operational"
        else:
            health_status["overall_health"] = "not_initialized"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Startup function for command-line usage
async def main():
    """Main function for running the CEO organization system."""
    try:
        logger.info("Starting CEO-Centric AI Organization System...")
        
        # Initialize with default business domain (can be customized via API)
        default_request = InitializeRequest(
            business_domain="Business Domain",
            industry="Software Development",
            business_context={
                "project_type": "Enterprise Application Development",
                "capabilities": {
                    "domain_skills": ["Business Analysis", "Process Optimization"],
                    "tools": ["Analytics Platforms", "Business Intelligence"]
                }
            }
        )
        
        # Initialize organization
        result = await initialize_organization(default_request)
        logger.info(f"Organization initialized: {result}")
        
        # Start the FastAPI server (if running directly)
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8503)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 