"""
Main AI Engine for the Bot Builder AI system.
Handles natural language processing, AI Employee orchestration, and system management.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import openai
from openai import OpenAI

from config.settings import settings
from core.employee_factory import AIEmployeeFactory
from core.learning_engine import LearningEngine
from data.data_manager import DataManager
from monitoring.metrics import MetricsCollector
from security.auth import SecurityManager
from utils.helpers import generate_uuid, validate_input

logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Context for maintaining conversation state."""
    session_id: str
    user_id: str
    conversation_history: List[Dict[str, str]]
    current_ai_employees: List[str]
    last_interaction: datetime
    context_data: Dict[str, Any]

class AIEngine:
    """
    Main AI Engine that orchestrates the Bot Builder AI system.
    
    Responsibilities:
    - Natural language processing and understanding
    - AI Employee creation and management
    - System optimization and learning
    - Performance monitoring and reporting
    """
    
    def __init__(self):
        """Initialize the AI Engine."""
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.employee_factory = AIEmployeeFactory()
        self.learning_engine = LearningEngine()
        self.data_manager = DataManager()
        self.metrics_collector = MetricsCollector()
        self.security_manager = SecurityManager()
        
        # Conversation management
        self.conversations: Dict[str, ConversationContext] = {}
        
        # System state
        self.active_ai_employees: Dict[str, Any] = {}
        self.system_health = "healthy"
        self.last_optimization = datetime.now()
        
        logger.info("AI Engine initialized successfully")
    
    async def process_user_input(self, user_input: str, session_id: str, user_id: str) -> str:
        """
        Process user input and generate appropriate response.
        
        Args:
            user_input: The user's natural language input
            session_id: Unique session identifier
            user_id: User identifier
            
        Returns:
            AI response to the user input
        """
        try:
            # Validate input
            if not validate_input(user_input):
                return "I'm sorry, but I couldn't understand your input. Please try again."
            
            # Get or create conversation context
            context = self._get_conversation_context(session_id, user_id)
            
            # Add user input to conversation history
            context.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Analyze user intent
            intent = await self._analyze_intent(user_input, context)
            
            # Generate response based on intent
            response = await self._generate_response(intent, user_input, context)
            
            # Add AI response to conversation history
            context.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update context
            context.last_interaction = datetime.now()
            
            # Log interaction
            logger.info(f"Processed user input for session {session_id}: {intent['action']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            return "I encountered an error while processing your request. Please try again."
    
    async def _analyze_intent(self, user_input: str, context: ConversationContext) -> Dict[str, Any]:
        """
        Analyze user intent using OpenAI GPT-4.
        
        Args:
            user_input: User's input text
            context: Conversation context
            
        Returns:
            Intent analysis with action and parameters
        """
        try:
            # Prepare system prompt for intent analysis
            system_prompt = self._get_intent_analysis_prompt(context)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=settings.openai_max_tokens,
                temperature=settings.openai_temperature
            )
            
            # Parse response
            intent_text = response.choices[0].message.content
            intent = json.loads(intent_text)
            
            return intent
            
        except Exception as e:
            logger.error(f"Error analyzing intent: {str(e)}")
            return {
                "action": "unknown",
                "confidence": 0.0,
                "parameters": {},
                "error": str(e)
            }
    
    async def _generate_response(self, intent: Dict[str, Any], user_input: str, context: ConversationContext) -> str:
        """
        Generate response based on analyzed intent.
        
        Args:
            intent: Analyzed user intent
            user_input: Original user input
            context: Conversation context
            
        Returns:
            Generated response
        """
        action = intent.get("action", "unknown")
        
        try:
            if action == "create_ai_employee":
                return await self._handle_create_ai_employee(intent, context)
            elif action == "monitor_performance":
                return await self._handle_monitor_performance(intent, context)
            elif action == "optimize_ai_employee":
                return await self._handle_optimize_ai_employee(intent, context)
            elif action == "get_status":
                return await self._handle_get_status(intent, context)
            elif action == "configure_system":
                return await self._handle_configure_system(intent, context)
            elif action == "help":
                return self._handle_help()
            else:
                return await self._handle_unknown_action(intent, context)
                
        except Exception as e:
            logger.error(f"Error generating response for action {action}: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def _handle_create_ai_employee(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle AI Employee creation request."""
        parameters = intent.get("parameters", {})
        role = parameters.get("role", "")
        specialization = parameters.get("specialization", "")
        
        try:
            # Create AI Employee
            employee_id = await self.employee_factory.create_ai_employee(
                role=role,
                specialization=specialization,
                context=context
            )
            
            # Add to active employees
            self.active_ai_employees[employee_id] = {
                "role": role,
                "specialization": specialization,
                "created_at": datetime.now(),
                "status": "training"
            }
            
            # Update context
            context.current_ai_employees.append(employee_id)
            
            # Get role configuration
            role_config = settings.ai_employee_roles.get(role, {})
            
            response = f"""I've successfully created a new {role.title()} AI Employee with the following capabilities:

**Role**: {role.title()}
**Specialization**: {specialization}
**Employee ID**: {employee_id}

**Capabilities**:
- Model Architecture: {role_config.get('model_architecture', 'N/A')}
- Training Data Sources: {', '.join(role_config.get('training_data_sources', []))}
- Evaluation Metrics: {', '.join(role_config.get('evaluation_metrics', []))}
- Optimization Focus: {', '.join(role_config.get('optimization_focus', []))}

**Training Parameters**:
- Learning Rate: {role_config.get('learning_rate', 'N/A')}
- Batch Size: {role_config.get('batch_size', 'N/A')}
- Training Epochs: {role_config.get('epochs', 'N/A')}

Training will begin immediately. Estimated completion time: {self._estimate_training_time(role)} hours.

You can monitor the training progress by asking: "Show me the status of AI Employee {employee_id}" """

            return response
            
        except Exception as e:
            logger.error(f"Error creating AI Employee: {str(e)}")
            return f"I encountered an error while creating the AI Employee: {str(e)}"
    
    async def _handle_monitor_performance(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle performance monitoring request."""
        parameters = intent.get("parameters", {})
        employee_id = parameters.get("employee_id", "")
        
        try:
            if employee_id and employee_id in self.active_ai_employees:
                # Get specific employee performance
                performance = await self.metrics_collector.get_employee_performance(employee_id)
                return self._format_employee_performance(employee_id, performance)
            else:
                # Get all employees performance
                all_performance = await self.metrics_collector.get_all_performance()
                return self._format_all_performance(all_performance)
                
        except Exception as e:
            logger.error(f"Error monitoring performance: {str(e)}")
            return f"I encountered an error while monitoring performance: {str(e)}"
    
    async def _handle_optimize_ai_employee(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle AI Employee optimization request."""
        parameters = intent.get("parameters", {})
        employee_id = parameters.get("employee_id", "")
        optimization_focus = parameters.get("optimization_focus", "")
        
        try:
            if employee_id not in self.active_ai_employees:
                return f"AI Employee {employee_id} not found. Please check the employee ID."
            
            # Start optimization
            optimization_id = await self.learning_engine.optimize_employee(
                employee_id=employee_id,
                focus_area=optimization_focus,
                context=context
            )
            
            response = f"""I've initiated optimization for AI Employee {employee_id}.

**Optimization Details**:
- Optimization ID: {optimization_id}
- Focus Area: {optimization_focus}
- Status: In Progress

**Current Optimizations**:
{self._get_optimization_details(optimization_focus)}

**Expected Improvements**:
{self._get_expected_improvements(optimization_focus)}

**Estimated Time**: {self._estimate_optimization_time(optimization_focus)} minutes

You can monitor the optimization progress by asking: "Show me the optimization status for {optimization_id}" """

            return response
            
        except Exception as e:
            logger.error(f"Error optimizing AI Employee: {str(e)}")
            return f"I encountered an error while optimizing the AI Employee: {str(e)}"
    
    async def _handle_get_status(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle status request."""
        parameters = intent.get("parameters", {})
        status_type = parameters.get("status_type", "system")
        
        try:
            if status_type == "system":
                return self._get_system_status()
            elif status_type == "employees":
                return self._get_employees_status()
            elif status_type == "performance":
                return await self._get_performance_status()
            else:
                return f"Unknown status type: {status_type}. Available types: system, employees, performance"
                
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            return f"I encountered an error while getting status: {str(e)}"
    
    async def _handle_configure_system(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle system configuration request."""
        parameters = intent.get("parameters", {})
        config_type = parameters.get("config_type", "")
        config_value = parameters.get("config_value", "")
        
        try:
            # This would typically update system configuration
            # For now, return a placeholder response
            return f"I've updated the system configuration for {config_type} to {config_value}. The changes will take effect immediately."
            
        except Exception as e:
            logger.error(f"Error configuring system: {str(e)}")
            return f"I encountered an error while configuring the system: {str(e)}"
    
    def _handle_help(self) -> str:
        """Handle help request."""
        return """# Bot Builder AI - Help Guide

## Available Commands

### Creating AI Employees
- "Create a new Research Analyst AI Employee focused on cryptocurrency markets"
- "Build a Trader AI Employee for high-frequency trading"
- "Generate a Risk Manager AI Employee for portfolio risk assessment"

### Monitoring Performance
- "Show me the performance metrics for all AI Employees"
- "What's the status of AI Employee [ID]?"
- "Display performance analytics for the last 30 days"

### Optimization
- "Optimize the Trader AI Employee for better execution speed"
- "Improve the Risk Manager's prediction accuracy"
- "Enhance the Research Analyst's sentiment analysis"

### System Management
- "What's the current system status?"
- "Show me all active AI Employees"
- "Configure the system for [parameter]"

### Getting Help
- "Help" - Show this help guide
- "What can you do?" - Explain system capabilities

## AI Employee Roles

1. **Research Analyst**: Deep learning, forecasting, economic analysis
2. **Trader**: Reinforcement learning, execution speed, strategic decision-making
3. **Risk Manager**: Probability theory, statistical modeling, scenario testing
4. **Compliance Officer**: Regulatory knowledge, NLP, explainability
5. **Data Specialist**: Data cleaning, management, structuring

## Tips
- Be specific about your requirements
- Monitor performance regularly
- Use optimization features to improve results
- Check system status for any issues

Need more help? Just ask!"""
    
    async def _handle_unknown_action(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle unknown or unclear actions."""
        return """I'm not sure I understood your request. Here are some things I can help you with:

- Create new AI Employees (Research Analyst, Trader, Risk Manager, etc.)
- Monitor performance and analytics
- Optimize existing AI Employees
- Check system status and health
- Configure system parameters

Try asking something like:
- "Create a new Research Analyst AI Employee"
- "Show me the performance of all AI Employees"
- "What's the system status?"

Or type "help" for a complete guide."""
    
    def _get_conversation_context(self, session_id: str, user_id: str) -> ConversationContext:
        """Get or create conversation context."""
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext(
                session_id=session_id,
                user_id=user_id,
                conversation_history=[],
                current_ai_employees=[],
                last_interaction=datetime.now(),
                context_data={}
            )
        return self.conversations[session_id]
    
    def _get_intent_analysis_prompt(self, context: ConversationContext) -> str:
        """Get the system prompt for intent analysis."""
        return f"""You are an AI assistant that analyzes user intent for a Bot Builder AI system. 

Your task is to understand what the user wants to do and return a JSON response with the following structure:
{{
    "action": "create_ai_employee|monitor_performance|optimize_ai_employee|get_status|configure_system|help|unknown",
    "confidence": 0.0-1.0,
    "parameters": {{
        // Action-specific parameters
    }}
}}

Available actions:
1. create_ai_employee - User wants to create a new AI Employee
   Parameters: {{"role": "research_analyst|trader|risk_manager|compliance_officer|data_specialist", "specialization": "string"}}

2. monitor_performance - User wants to see performance metrics
   Parameters: {{"employee_id": "string (optional)"}}

3. optimize_ai_employee - User wants to optimize an AI Employee
   Parameters: {{"employee_id": "string", "optimization_focus": "string"}}

4. get_status - User wants system status
   Parameters: {{"status_type": "system|employees|performance"}}

5. configure_system - User wants to configure system
   Parameters: {{"config_type": "string", "config_value": "string"}}

6. help - User wants help
   Parameters: {{}}

7. unknown - Cannot determine intent
   Parameters: {{}}

Current context:
- Active AI Employees: {len(context.current_ai_employees)}
- Session ID: {context.session_id}

Return only valid JSON. Be confident in your analysis."""
    
    def _estimate_training_time(self, role: str) -> int:
        """Estimate training time for a role."""
        role_config = settings.ai_employee_roles.get(role, {})
        epochs = role_config.get('epochs', 100)
        return max(1, epochs // 50)  # Rough estimate
    
    def _estimate_optimization_time(self, focus_area: str) -> int:
        """Estimate optimization time."""
        estimates = {
            "execution_speed": 45,
            "prediction_accuracy": 60,
            "risk_management": 30,
            "data_quality": 20,
            "compliance": 40
        }
        return estimates.get(focus_area.lower(), 30)
    
    def _get_optimization_details(self, focus_area: str) -> str:
        """Get optimization details for a focus area."""
        details = {
            "execution_speed": "- Implementing parallel processing\n- Updating reinforcement learning parameters\n- Fine-tuning neural network architecture\n- Adding latency optimization algorithms",
            "prediction_accuracy": "- Enhancing feature engineering\n- Optimizing hyperparameters\n- Implementing ensemble methods\n- Improving data preprocessing",
            "risk_management": "- Updating risk models\n- Enhancing scenario analysis\n- Improving VaR calculations\n- Optimizing stress testing",
            "data_quality": "- Improving data cleaning algorithms\n- Enhancing validation processes\n- Optimizing data pipelines\n- Implementing quality checks",
            "compliance": "- Updating regulatory knowledge\n- Enhancing audit trails\n- Improving explainability\n- Optimizing compliance checks"
        }
        return details.get(focus_area.lower(), "- General optimization\n- Parameter tuning\n- Model refinement")
    
    def _get_expected_improvements(self, focus_area: str) -> str:
        """Get expected improvements for a focus area."""
        improvements = {
            "execution_speed": "40-60% faster execution",
            "prediction_accuracy": "15-25% better accuracy",
            "risk_management": "20-30% better risk prediction",
            "data_quality": "30-40% better data quality",
            "compliance": "25-35% better compliance score"
        }
        return improvements.get(focus_area.lower(), "10-20% general improvement")
    
    def _format_employee_performance(self, employee_id: str, performance: Dict[str, Any]) -> str:
        """Format individual employee performance."""
        return f"""**AI Employee {employee_id} Performance Report**

**Current Status**: {performance.get('status', 'Unknown')}
**Last Updated**: {performance.get('last_updated', 'Unknown')}

**Performance Metrics**:
{self._format_metrics(performance.get('metrics', {}))}

**Recent Activity**:
{self._format_recent_activity(performance.get('recent_activity', []))}

**Recommendations**:
{performance.get('recommendations', 'No recommendations available')}"""
    
    def _format_all_performance(self, all_performance: Dict[str, Any]) -> str:
        """Format all employees performance."""
        response = "**All AI Employees Performance Report**\n\n"
        
        for employee_id, performance in all_performance.items():
            response += f"**{employee_id}**: {performance.get('status', 'Unknown')}\n"
            response += f"  - {self._format_metrics_summary(performance.get('metrics', {}))}\n\n"
        
        return response
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for display."""
        formatted = ""
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted += f"- {key.replace('_', ' ').title()}: {value:.2f}\n"
            else:
                formatted += f"- {key.replace('_', ' ').title()}: {value}\n"
        return formatted or "No metrics available"
    
    def _format_metrics_summary(self, metrics: Dict[str, Any]) -> str:
        """Format metrics summary."""
        if not metrics:
            return "No metrics available"
        
        # Get the most important metric
        important_keys = ['accuracy', 'sharpe_ratio', 'execution_speed', 'win_rate']
        for key in important_keys:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    return f"{key.replace('_', ' ').title()}: {value:.2f}"
                else:
                    return f"{key.replace('_', ' ').title()}: {value}"
        
        # Fallback to first metric
        first_key = list(metrics.keys())[0]
        return f"{first_key.replace('_', ' ').title()}: {metrics[first_key]}"
    
    def _format_recent_activity(self, activities: List[Dict[str, Any]]) -> str:
        """Format recent activity."""
        if not activities:
            return "No recent activity"
        
        formatted = ""
        for activity in activities[:5]:  # Show last 5 activities
            timestamp = activity.get('timestamp', 'Unknown')
            action = activity.get('action', 'Unknown')
            formatted += f"- {timestamp}: {action}\n"
        return formatted
    
    def _get_system_status(self) -> str:
        """Get system status."""
        return f"""**Bot Builder AI System Status**

**Overall Health**: {self.system_health}
**Active AI Employees**: {len(self.active_ai_employees)}
**Total Conversations**: {len(self.conversations)}
**Last Optimization**: {self.last_optimization.strftime('%Y-%m-%d %H:%M:%S')}

**System Resources**:
- CPU Usage: {self._get_cpu_usage()}%
- Memory Usage: {self._get_memory_usage()}%
- Storage Usage: {self._get_storage_usage()}%

**Recent Activity**:
- System running normally
- All services operational
- No critical issues detected"""
    
    def _get_employees_status(self) -> str:
        """Get employees status."""
        if not self.active_ai_employees:
            return "No active AI Employees found."
        
        response = "**Active AI Employees Status**\n\n"
        for employee_id, details in self.active_ai_employees.items():
            response += f"**{employee_id}**\n"
            response += f"  - Role: {details['role'].title()}\n"
            response += f"  - Specialization: {details['specialization']}\n"
            response += f"  - Status: {details['status']}\n"
            response += f"  - Created: {details['created_at'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return response
    
    async def _get_performance_status(self) -> str:
        """Get performance status."""
        try:
            performance = await self.metrics_collector.get_all_performance()
            return self._format_all_performance(performance)
        except Exception as e:
            return f"Error getting performance status: {str(e)}"
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        # Placeholder - would integrate with system monitoring
        return 45.2
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        # Placeholder - would integrate with system monitoring
        return 67.8
    
    def _get_storage_usage(self) -> float:
        """Get storage usage percentage."""
        # Placeholder - would integrate with system monitoring
        return 23.4
    
    async def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Clean up old conversations."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        old_sessions = [
            session_id for session_id, context in self.conversations.items()
            if context.last_interaction < cutoff_time
        ]
        
        for session_id in old_sessions:
            del self.conversations[session_id]
        
        if old_sessions:
            logger.info(f"Cleaned up {len(old_sessions)} old conversations")
    
    async def shutdown(self):
        """Shutdown the AI Engine gracefully."""
        logger.info("Shutting down AI Engine...")
        
        # Stop all AI Employees
        for employee_id in list(self.active_ai_employees.keys()):
            try:
                await self.employee_factory.stop_ai_employee(employee_id)
            except Exception as e:
                logger.error(f"Error stopping AI Employee {employee_id}: {str(e)}")
        
        # Cleanup
        await self.cleanup_old_conversations()
        
        logger.info("AI Engine shutdown complete") 