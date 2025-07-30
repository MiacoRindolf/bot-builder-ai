"""
Intelligent Chat Interface - Smart AI Assistant for Bot Builder System
Routes between local and cloud LLMs based on task complexity.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import re

from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    """Task complexity levels for routing."""
    SIMPLE = "simple"      # Local LLM
    MEDIUM = "medium"      # Local LLM with fallback
    COMPLEX = "complex"    # OpenAI
    CRITICAL = "critical"  # OpenAI only

@dataclass
class ChatMessage:
    """A chat message with metadata."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    task_complexity: Optional[TaskComplexity] = None
    processing_time: Optional[float] = None
    llm_used: Optional[str] = None

@dataclass
class SystemStatus:
    """Real-time system status for chat context."""
    timestamp: datetime
    active_employees: int
    pending_proposals: int
    system_health: float
    recent_activities: List[str]
    current_tasks: List[str]
    performance_metrics: Dict[str, Any]

class IntelligentChatInterface:
    """
    Intelligent Chat Interface that routes between local and cloud LLMs.
    
    Features:
    - Smart task routing based on complexity
    - Real-time system status integration
    - Conversational AI assistant capabilities
    - System navigation and task delegation
    """
    
    def __init__(self, ai_engine):
        """Initialize the chat interface."""
        self.ai_engine = ai_engine
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Chat history
        self.chat_history: List[ChatMessage] = []
        self.max_history = 50
        
        # System status tracking
        self.last_status_update = None
        self.status_cache_duration = 30  # seconds
        
        # Task complexity patterns
        self.complexity_patterns = {
            TaskComplexity.SIMPLE: [
                r"show|display|list|get|what|how many|status|dashboard",
                r"navigate|go to|open|switch to",
                r"help|assist|guide",
                r"simple|basic|quick"
            ],
            TaskComplexity.MEDIUM: [
                r"explain|describe|tell me about|analyze|review",
                r"compare|difference|similar|versus",
                r"create|add|new|generate",
                r"modify|change|update|edit"
            ],
            TaskComplexity.COMPLEX: [
                r"optimize|improve|enhance|upgrade",
                r"analyze.*performance|system.*analysis",
                r"generate.*proposal|create.*improvement",
                r"debug|troubleshoot|fix.*issue"
            ],
            TaskComplexity.CRITICAL: [
                r"critical|urgent|emergency|fix.*now",
                r"system.*failure|error.*critical",
                r"security|vulnerability|breach",
                r"production.*issue|live.*problem"
            ]
        }
        
        # Local LLM configuration (placeholder for DeepSeek)
        self.local_llm_available = False
        self.local_llm_model = "deepseek-coder:6.7b"  # Example model name
        
        logger.info("Intelligent Chat Interface initialized")
    
    async def initialize(self) -> bool:
        """Initialize the chat interface."""
        try:
            # Check if local LLM is available
            self.local_llm_available = await self._check_local_llm_availability()
            
            # Initialize system status
            await self._update_system_status()
            
            logger.info(f"Chat Interface initialized. Local LLM: {'Available' if self.local_llm_available else 'Not available'}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chat interface: {str(e)}")
            return False
    
    async def process_message(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        
        Args:
            user_message: The user's message
            
        Returns:
            The AI assistant's response
        """
        try:
            start_time = time.time()
            
            # Analyze task complexity
            complexity = self._analyze_task_complexity(user_message)
            
            # Update system status for context
            await self._update_system_status()
            
            # Route to appropriate LLM
            if complexity in [TaskComplexity.SIMPLE, TaskComplexity.MEDIUM] and self.local_llm_available:
                response = await self._process_with_local_llm(user_message, complexity)
                llm_used = "local"
            else:
                response = await self._process_with_openai(user_message, complexity)
                llm_used = "openai"
            
            processing_time = time.time() - start_time
            
            # Record the conversation
            self._add_to_history("user", user_message)
            self._add_to_history("assistant", response, complexity, processing_time, llm_used)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    def _analyze_task_complexity(self, message: str) -> TaskComplexity:
        """Analyze the complexity of a user message."""
        message_lower = message.lower()
        
        # Check patterns from most complex to least
        for complexity in [TaskComplexity.CRITICAL, TaskComplexity.COMPLEX, TaskComplexity.MEDIUM, TaskComplexity.SIMPLE]:
            patterns = self.complexity_patterns[complexity]
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return complexity
        
        # Default to medium complexity
        return TaskComplexity.MEDIUM
    
    async def _process_with_local_llm(self, message: str, complexity: TaskComplexity) -> str:
        """Process message using local DeepSeek LLM."""
        try:
            # This is a placeholder for local LLM integration
            # In practice, you'd use ollama, llama.cpp, or similar
            
            system_prompt = self._create_system_prompt(complexity)
            
            # Simulate local LLM response (replace with actual implementation)
            if "status" in message.lower() or "dashboard" in message.lower():
                return await self._get_system_status_response()
            elif "proposals" in message.lower() or "pending" in message.lower():
                return await self._get_proposals_response()
            elif "employees" in message.lower() or "ai employees" in message.lower():
                return await self._get_employees_response()
            else:
                return f"I'm your local AI assistant. I can help you with system navigation, status checks, and basic tasks. For complex analysis, I'll route to the cloud AI. What would you like to know about the system?"
                
        except Exception as e:
            logger.error(f"Error with local LLM: {str(e)}")
            # Fallback to OpenAI
            return await self._process_with_openai(message, complexity)
    
    async def _process_with_openai(self, message: str, complexity: TaskComplexity) -> str:
        """Process message using OpenAI."""
        try:
            system_prompt = self._create_system_prompt(complexity)
            
            # Get recent context
            recent_context = self._get_recent_context()
            
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Recent context: {recent_context}\n\nUser message: {message}"}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error with OpenAI: {str(e)}")
            return f"I apologize, but I'm having trouble connecting to the AI service. Please try again later."
    
    def _create_system_prompt(self, complexity: TaskComplexity) -> str:
        """Create a system prompt based on task complexity."""
        base_prompt = f"""
You are an intelligent AI assistant for the Bot Builder system. You help users navigate, understand, and interact with the system.

Current system status:
{self._format_system_status()}

Your capabilities:
- System navigation and status reporting
- Task delegation and execution
- Real-time system monitoring
- Self-improvement proposal management
- AI employee management
- Technical analysis and recommendations

Task complexity: {complexity.value}

Guidelines:
- Be helpful, professional, and concise
- Provide actionable information
- Use the system's real data when available
- Offer to help with specific tasks
- Explain what the system is currently doing
"""

        if complexity == TaskComplexity.CRITICAL:
            base_prompt += "\n\nCRITICAL: This is a high-priority task. Provide immediate, actionable responses."
        elif complexity == TaskComplexity.COMPLEX:
            base_prompt += "\n\nCOMPLEX: This requires deep analysis. Provide detailed, well-reasoned responses."
        
        return base_prompt
    
    def _format_system_status(self) -> str:
        """Format current system status for the prompt."""
        if not self.last_status_update:
            return "System status: Initializing..."
        
        status = self.last_status_update
        return f"""
System Health: {status.system_health:.1%}
Active AI Employees: {status.active_employees}
Pending Proposals: {status.pending_proposals}
Recent Activities: {', '.join(status.recent_activities[-3:])}
Current Tasks: {', '.join(status.current_tasks[-3:])}
"""
    
    async def _get_system_status_response(self) -> str:
        """Get a formatted system status response."""
        if not self.last_status_update:
            return "🔄 System is initializing. Please wait a moment..."
        
        status = self.last_status_update
        return f"""
📊 **System Status Overview**

🏥 **Health Score:** {status.system_health:.1%}
👥 **Active AI Employees:** {status.active_employees}
📋 **Pending Proposals:** {status.pending_proposals}

🔄 **Recent Activities:**
{chr(10).join(f"• {activity}" for activity in status.recent_activities[-3:])}

⚡ **Current Tasks:**
{chr(10).join(f"• {task}" for task in status.current_tasks[-3:])}

💡 **Quick Actions:**
• Type "show proposals" to see pending improvements
• Type "show employees" to see AI team status
• Type "analyze system" for detailed analysis
"""
    
    async def _get_proposals_response(self) -> str:
        """Get a formatted proposals response."""
        try:
            proposals = await self.ai_engine.self_improvement_engine.get_pending_proposals()
            
            if not proposals:
                return "✅ No pending proposals. The system is running optimally!"
            
            response = f"📋 **Pending Proposals ({len(proposals)})**\n\n"
            
            for i, proposal in enumerate(proposals[:5], 1):
                response += f"{i}. **{proposal.title}**\n"
                response += f"   📝 {proposal.description[:100]}...\n"
                response += f"   🎯 Impact: {proposal.estimated_impact.get('performance_improvement', 'Medium')}\n"
                response += f"   ⚠️ Risk: {proposal.risk_level}\n\n"
            
            if len(proposals) > 5:
                response += f"... and {len(proposals) - 5} more proposals\n\n"
            
            response += "💡 **Actions:**\n"
            response += "• Go to Self-Improvement Hub to approve/reject\n"
            response += "• Type 'analyze system' for detailed analysis\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error retrieving proposals: {str(e)}"
    
    async def _get_employees_response(self) -> str:
        """Get a formatted employees response."""
        try:
            # Get employee data from metrics
            employee_data = await self.ai_engine.metrics_collector.get_employee_performance()
            
            if not employee_data:
                return "👥 No AI employees currently active."
            
            response = "👥 **AI Employee Status**\n\n"
            
            for employee in employee_data[:5]:
                name = employee.get('name', 'Unknown')
                role = employee.get('role', 'Unknown')
                status = employee.get('status', 'Unknown')
                performance = employee.get('performance_score', 0)
                
                response += f"**{name}** ({role})\n"
                response += f"   📊 Status: {status}\n"
                response += f"   🎯 Performance: {performance:.1%}\n\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error retrieving employee data: {str(e)}"
    
    async def _update_system_status(self) -> None:
        """Update the cached system status."""
        try:
            # Only update if cache is expired
            if (self.last_status_update and 
                (datetime.now() - self.last_status_update.timestamp).seconds < self.status_cache_duration):
                return
            
            # Get real system data
            metrics = await self.ai_engine.metrics_collector.get_system_metrics()
            proposals = await self.ai_engine.self_improvement_engine.get_pending_proposals()
            
            # Create status object
            self.last_status_update = SystemStatus(
                timestamp=datetime.now(),
                active_employees=metrics.get('total_ai_employees', 0),
                pending_proposals=len(proposals),
                system_health=metrics.get('system_health', 0.8),
                recent_activities=self._get_recent_activities(),
                current_tasks=self._get_current_tasks(),
                performance_metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error updating system status: {str(e)}")
    
    def _get_recent_activities(self) -> List[str]:
        """Get recent system activities."""
        activities = []
        
        # Add recent chat messages
        recent_messages = self.chat_history[-5:]
        for msg in recent_messages:
            if msg.role == "user":
                activities.append(f"User asked: {msg.content[:50]}...")
        
        # Add system activities (placeholder)
        activities.extend([
            "System analysis completed",
            "Performance metrics updated",
            "AI employees active"
        ])
        
        return activities[-5:]  # Return last 5
    
    def _get_current_tasks(self) -> List[str]:
        """Get current system tasks."""
        tasks = []
        
        # Check for pending proposals
        if hasattr(self.ai_engine, 'self_improvement_engine'):
            try:
                proposals = asyncio.run(self.ai_engine.self_improvement_engine.get_pending_proposals())
                if proposals:
                    tasks.append(f"Processing {len(proposals)} pending proposals")
            except:
                pass
        
        # Add default tasks
        tasks.extend([
            "Monitoring system performance",
            "Managing AI employees",
            "Collecting metrics"
        ])
        
        return tasks[-3:]  # Return last 3
    
    def _get_recent_context(self) -> str:
        """Get recent conversation context."""
        recent_messages = self.chat_history[-6:]  # Last 6 messages (3 exchanges)
        
        context = []
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            context.append(f"{role}: {msg.content}")
        
        return "\n".join(context)
    
    def _add_to_history(self, role: str, content: str, complexity: Optional[TaskComplexity] = None,
                       processing_time: Optional[float] = None, llm_used: Optional[str] = None) -> None:
        """Add a message to the chat history."""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            task_complexity=complexity,
            processing_time=processing_time,
            llm_used=llm_used
        )
        
        self.chat_history.append(message)
        
        # Keep history within limit
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]
    
    async def _check_local_llm_availability(self) -> bool:
        """Check if local LLM is available."""
        try:
            # This is a placeholder - implement actual local LLM check
            # For now, return False to use OpenAI
            return False
            
        except Exception as e:
            logger.error(f"Error checking local LLM availability: {str(e)}")
            return False
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get formatted chat history for UI."""
        history = []
        
        for msg in self.chat_history:
            history.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "complexity": msg.task_complexity.value if msg.task_complexity else None,
                "processing_time": msg.processing_time,
                "llm_used": msg.llm_used
            })
        
        return history
    
    def clear_history(self) -> None:
        """Clear chat history."""
        self.chat_history = []
        logger.info("Chat history cleared") 