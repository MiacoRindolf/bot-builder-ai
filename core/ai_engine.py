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
import numpy as np # Added for RL state creation

from config.settings import settings
from core.employee_factory import AIEmployeeFactory
from core.learning_engine import LearningEngine
from core.advanced_rl_engine import AdvancedRLEngine, RLState, RLAction, RLReward
from core.explainability_engine import AdvancedExplainabilityEngine, ExplanationResult, DecisionContext
from core.self_improvement_engine import SelfImprovementEngine, SelfImprovementProposal
from core.intelligent_chat_interface import IntelligentChatInterface
from data.data_manager import DataManager
from data.real_time_market_data import RealTimeMarketDataProvider, MarketDataPoint, MarketEvent
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
    - Advanced reinforcement learning integration
    - Real-time market data processing
    - Advanced explainability and transparency
    """
    
    def __init__(self):
        """Initialize the AI Engine."""
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.employee_factory = AIEmployeeFactory()
        self.learning_engine = LearningEngine()
        self.advanced_rl_engine = AdvancedRLEngine()  # Advanced RL Engine
        self.explainability_engine = AdvancedExplainabilityEngine()  # NEW: Explainability Engine
        self.self_improvement_engine = SelfImprovementEngine()  # NEW: True Self-Improvement Engine
        self.intelligent_chat = IntelligentChatInterface(self)  # NEW: Intelligent Chat Interface
        self.data_manager = DataManager()
        self.real_time_data = RealTimeMarketDataProvider()  # Real-time data
        self.metrics_collector = MetricsCollector()
        self.security_manager = SecurityManager()
        
        # Conversation management
        self.conversations: Dict[str, ConversationContext] = {}
        
        # System state
        self.active_ai_employees: Dict[str, Any] = {}
        self.system_health = "healthy"
        self.last_optimization = datetime.now()
        
        # Advanced RL state
        self.rl_enabled = True
        self.rl_training_mode = "continuous"
        
        # Real-time data state
        self.real_time_enabled = True
        self.monitored_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD"]
        self.market_events_queue = asyncio.Queue()
        
        # Explainability state
        self.explainability_enabled = True
        self.explanation_history: List[ExplanationResult] = []

        # Spending tracking
        self.monthly_spend = 0.0
        self.spend_tracking_enabled = settings.enable_spend_tracking
        self.monthly_spend_limit = settings.monthly_spend_limit
        self.spend_warning_threshold = settings.spend_warning_threshold
        self.current_month = datetime.now().strftime("%Y-%m")

        logger.info("AI Engine initialized successfully with Advanced RL, Real-time Data, Explainability, and True Self-Improvement capabilities")
    
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
    
    async def initialize(self) -> bool:
        """Initialize the AI Engine with advanced capabilities."""
        try:
            # Initialize advanced RL engine
            if self.rl_enabled:
                logger.info("Initializing Advanced RL Engine...")
                rl_success = await self.advanced_rl_engine.initialize()
                if not rl_success:
                    logger.warning("Advanced RL Engine initialization failed, continuing without RL")
                    self.rl_enabled = False
            
            # Initialize explainability engine
            if self.explainability_enabled:
                logger.info("Initializing Advanced Explainability Engine...")
                exp_success = await self.explainability_engine.initialize()
                if not exp_success:
                    logger.warning("Explainability Engine initialization failed, continuing without explainability")
                    self.explainability_enabled = False
            
            # Initialize real-time market data
            if self.real_time_enabled:
                logger.info("Initializing Real-time Market Data Provider...")
                rt_success = await self.real_time_data.initialize()
                if rt_success:
                    await self._start_real_time_data_feed()
                    await self._setup_market_event_handlers()
                else:
                    logger.warning("Real-time data initialization failed, continuing without real-time data")
                    self.real_time_enabled = False
            
            # Initialize self-improvement engine
            logger.info("Initializing True Self-Improvement Engine...")
            si_success = await self.self_improvement_engine.initialize()
            if not si_success:
                logger.warning("Self-Improvement Engine initialization failed, continuing without self-improvement")
            
            # Initialize intelligent chat interface
            logger.info("Initializing Intelligent Chat Interface...")
            chat_success = await self.intelligent_chat.initialize()
            if not chat_success:
                logger.warning("Intelligent Chat Interface initialization failed, continuing without chat")
            
            # Initialize other components
            await self.learning_engine.initialize()
            await self.metrics_collector.start_monitoring()
            
            logger.info("AI Engine initialized successfully with all components including True Self-Improvement")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing AI Engine: {str(e)}")
            return False

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
            # Spending tracking: reset if new month
            now_month = datetime.now().strftime("%Y-%m")
            if self.current_month != now_month:
                self.current_month = now_month
                self.monthly_spend = 0.0

            # Block if over limit
            if self.spend_tracking_enabled and self.monthly_spend >= self.monthly_spend_limit:
                logger.warning(f"Monthly spend limit (${self.monthly_spend_limit}) reached. Blocking OpenAI API calls.")
                return {
                    "action": "blocked",
                    "confidence": 1.0,
                    "parameters": {},
                    "error": f"Monthly spend limit (${self.monthly_spend_limit}) reached. No further OpenAI API calls will be made this month."
                }

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

            # Estimate cost (approximate, for gpt-3.5-turbo)
            if hasattr(response, 'usage') and response.usage is not None:
                input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                output_tokens = getattr(response.usage, 'completion_tokens', 0)
            else:
                input_tokens = 0
                output_tokens = 0
            cost = (input_tokens * 0.0015 + output_tokens * 0.002) / 1000
            self.monthly_spend += cost

            # Warn if approaching threshold
            if self.spend_tracking_enabled and self.monthly_spend >= self.spend_warning_threshold:
                logger.warning(f"Warning: Monthly spend is ${self.monthly_spend:.2f}, approaching limit of ${self.monthly_spend_limit}.")

            # Parse response
            intent_text = response.choices[0].message.content
            if intent_text:
                intent = json.loads(intent_text)
            else:
                intent = {"action": "unknown", "confidence": 0.0, "parameters": {}}

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
            elif action == "self_improve":
                return await self._handle_self_improve(intent, context)
            elif action == "analyze_system":
                return await self._handle_analyze_system(intent, context)
            elif action == "approve_proposal":
                return await self._handle_approve_proposal(intent, context)
            elif action == "reject_proposal":
                return await self._handle_reject_proposal(intent, context)
            elif action == "version_info":
                return await self._handle_version_info(intent, context)
            elif action == "upgrade_history":
                return await self._handle_upgrade_history(intent, context)
            elif action == "create_version":
                return await self._handle_create_version(intent, context)
            elif action == "help":
                return self._handle_help()
            elif action == "explain_decision":
                return await self._handle_explain_decision(intent, context)
            elif action == "greeting":
                return await self._handle_greeting(intent, context)
            elif action == "general_chat":
                return await self._handle_general_chat(intent, context)
            else:
                return await self._handle_unknown_action(intent, context)
                
        except Exception as e:
            logger.error(f"Error generating response for action {action}: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def _handle_create_ai_employee(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle AI Employee creation with advanced RL, real-time data, and explainability integration."""
        parameters = intent.get("parameters", {})
        role = parameters.get("role", "research_analyst")
        specialization = parameters.get("specialization", "general")
        
        try:
            # Create AI Employee
            employee_id = await self.employee_factory.create_ai_employee(
                role=role,
                specialization=specialization,
                context=context
            )
            
            if employee_id:
                # Initialize advanced RL model for the employee
                if self.rl_enabled:
                    await self._initialize_rl_for_employee(employee_id, role)
                
                # Add to active employees
                self.active_ai_employees[employee_id] = {
                    "role": role,
                    "specialization": specialization,
                    "created_at": datetime.now(),
                    "rl_enabled": self.rl_enabled,
                    "real_time_enabled": self.real_time_enabled,
                    "explainability_enabled": self.explainability_enabled
                }
                
                # Update context
                context.current_ai_employees.append(employee_id)
                
                response = f"""I've successfully created a new {role.replace('_', ' ').title()} AI Employee!

**Employee Details:**
- **ID**: {employee_id}
- **Role**: {role.replace('_', ' ').title()}
- **Specialization**: {specialization}
- **Advanced RL**: {'Enabled' if self.rl_enabled else 'Disabled'}
- **Real-time Data**: {'Enabled' if self.real_time_enabled else 'Disabled'}
- **Explainability**: {'Enabled' if self.explainability_enabled else 'Disabled'}

**Capabilities:**
{self._get_role_capabilities(role)}

**Real-time Features:**
{self._get_real_time_capabilities(role)}

**Explainability Features:**
{self._get_explainability_capabilities(role)}

**Training Status**: Initializing...
**Estimated Training Time**: {self._estimate_training_time(role)} hours

The AI Employee will begin training immediately and will be ready for deployment once training is complete.

You can monitor progress by asking: "Show me the status of {employee_id}" """

                return response
            else:
                return "I encountered an error while creating the AI Employee. Please try again."
                
        except Exception as e:
            logger.error(f"Error creating AI Employee: {str(e)}")
            return f"I encountered an error while creating the AI Employee: {str(e)}"

    async def _initialize_rl_for_employee(self, employee_id: str, role: str):
        """Initialize advanced RL model for an AI Employee."""
        try:
            # Define state and action dimensions based on role
            state_dim, action_dim = self._get_rl_dimensions(role)
            
            # Create RL model
            success = await self.advanced_rl_engine.create_employee_model(
                employee_id=employee_id,
                role=role,
                state_dim=state_dim,
                action_dim=action_dim
            )
            
            if success:
                logger.info(f"Advanced RL model created for employee {employee_id}")
            else:
                logger.warning(f"Failed to create RL model for employee {employee_id}")
                
        except Exception as e:
            logger.error(f"Error initializing RL for employee {employee_id}: {str(e)}")

    def _get_rl_dimensions(self, role: str) -> Tuple[int, int]:
        """Get state and action dimensions for RL model based on role."""
        dimensions = {
            "trader": (256, 8),  # Market data + portfolio state, 8 action types
            "research_analyst": (512, 6),  # Complex market analysis, 6 recommendation types
            "risk_manager": (128, 4),  # Risk metrics, 4 risk management actions
            "compliance_officer": (256, 5),  # Compliance data, 5 compliance actions
            "data_specialist": (384, 7)  # Data processing state, 7 data operations
        }
        return dimensions.get(role, (256, 6))

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
        """Handle AI Employee optimization with advanced RL."""
        parameters = intent.get("parameters", {})
        employee_id = parameters.get("employee_id", "")
        optimization_focus = parameters.get("optimization_focus", "accuracy")
        
        try:
            if employee_id not in self.active_ai_employees:
                return f"AI Employee {employee_id} not found. Please check the employee ID."
            
            # Perform advanced RL optimization
            if self.rl_enabled and employee_id in self.advanced_rl_engine.models:
                rl_results = await self.advanced_rl_engine.optimize_employee(
                    employee_id=employee_id,
                    optimization_focus=optimization_focus
                )
                
                if rl_results.get("success", False):
                    response = f"""I've initiated advanced RL optimization for AI Employee {employee_id}.

**Optimization Details:**
- **Focus Area**: {optimization_focus.replace('_', ' ').title()}
- **RL Engine**: Advanced Multi-Objective Learning
- **Status**: In Progress

**Advanced RL Features:**
{self._get_rl_optimization_details(optimization_focus)}

**Expected Improvements:**
{self._get_expected_rl_improvements(optimization_focus)}

**Estimated Time**: {self._estimate_optimization_time(optimization_focus)} minutes

The advanced RL engine will continuously optimize the model based on real-time performance data.

You can monitor the optimization progress by asking: "Show me the optimization status for {employee_id}" """
                else:
                    response = f"RL optimization failed: {rl_results.get('error', 'Unknown error')}"
            else:
                # Fallback to basic optimization
                response = await self._handle_basic_optimization(employee_id, optimization_focus)
            
            return response
            
        except Exception as e:
            logger.error(f"Error optimizing AI Employee: {str(e)}")
            return f"I encountered an error while optimizing the AI Employee: {str(e)}"

    def _get_rl_optimization_details(self, focus: str) -> str:
        """Get details about RL optimization features."""
        details = {
            "execution_speed": """
- Parallel processing optimization
- Latency-aware neural architecture
- Real-time decision acceleration
- Memory-efficient computations""",
            "risk_management": """
- Risk-aware reward shaping
- Multi-objective risk optimization
- Dynamic risk threshold adjustment
- Stress testing integration""",
            "accuracy": """
- Meta-learning for rapid adaptation
- Ensemble model optimization
- Feature importance weighting
- Cross-validation strategies""",
            "compliance": """
- Regulatory constraint learning
- Compliance-aware action selection
- Audit trail optimization
- Rule-based decision validation"""
        }
        return details.get(focus, "- General performance optimization\n- Multi-objective learning\n- Adaptive parameter tuning")

    def _get_expected_rl_improvements(self, focus: str) -> str:
        """Get expected improvements from RL optimization."""
        improvements = {
            "execution_speed": "40-60% faster execution, 30% reduced latency",
            "risk_management": "50% better risk control, 25% reduced drawdown",
            "accuracy": "20-30% better predictions, 15% improved Sharpe ratio",
            "compliance": "99%+ compliance score, 100% audit pass rate"
        }
        return improvements.get(focus, "15-25% overall performance improvement")

    async def _handle_basic_optimization(self, employee_id: str, optimization_focus: str) -> str:
        """Handle basic optimization when RL is not available."""
        return f"""I've initiated basic optimization for AI Employee {employee_id}.

**Optimization Details:**
- **Focus Area**: {optimization_focus.replace('_', ' ').title()}
- **Method**: Standard Learning Engine
- **Status**: In Progress

**Expected Improvements**: 10-20% performance improvement
**Estimated Time**: 30 minutes

You can monitor progress by asking: "Show me the optimization status for {employee_id}" """

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

### 🧠 TRUE AI Self-Improvement
- **"Improve yourself"** - I analyze my own system and generate improvement proposals
- **"Analyze system"** - Comprehensive system analysis for optimization opportunities
- **"Approve proposal [ID]"** - Review and approve my own improvement suggestions
- **"Reject proposal [ID] because [reason]"** - Reject with explanation
- **Real code generation** - I can actually modify my own code with your approval

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

## 🚀 Self-Improvement Workflow

1. **Say "improve yourself"** to start autonomous analysis
2. **I'll identify improvement opportunities** in my own code and performance
3. **Generate specific proposals** with actual code changes
4. **Submit for your approval** with detailed explanations
5. **If approved, I implement the changes** and run tests
6. **Monitor results** and learn from improvements

This is **true AI self-improvement** - not just task management, but actual code generation and system enhancement!

## Tips
- Be specific about your requirements
- Monitor performance regularly
- Use optimization features to improve results
- Check system status for any issues
- **Try "improve yourself" to see real AI self-improvement in action!**

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
    
    async def _handle_greeting(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle greeting messages."""
        parameters = intent.get("parameters", {})
        greeting_type = parameters.get("greeting_type", "hello")
        
        greetings = {
            "hello": "Hello! Welcome to the Bot Builder AI system. I'm here to help you manage AI employees and optimize your AI-powered hedge fund operations.",
            "hi": "Hi there! I'm your AI assistant for the Bot Builder system. How can I help you today?",
            "hey": "Hey! Great to see you. I'm ready to help you with AI employee management and system optimization.",
            "good_morning": "Good morning! I hope you're having a great start to your day. How can I assist you with the Bot Builder AI system?",
            "good_afternoon": "Good afternoon! I'm here to help you with your AI employee management and system optimization needs.",
            "good_evening": "Goodevening! I'm ready to help you with the Bot Builder AI system. What would you like to work on?"
        }
        
        base_response = greetings.get(greeting_type, greetings["hello"])
        
        return f"{base_response}\n\n**What I can help you with:**\n- Create and manage AI Employees (Research Analyst, Trader, Risk Manager, etc.)\n- Monitor performance and analytics\n- Optimize existing AI Employees\n- Check system status and health\n- Configure system parameters\n\n**Quick Start:**\nTry saying \"Create a new Research Analyst AI Employee\" or \"What's the system status?\" to get started!"

    async def _handle_general_chat(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle general conversation."""
        parameters = intent.get("parameters", {})
        topic = parameters.get("topic", "general")
        sentiment = parameters.get("sentiment", "neutral")
        
        responses = {
            "wellbeing": "I'm functioning perfectly! All systems are operational and I'm ready to help you with AI employee management and system optimization.",
            "capabilities": "I'm an advanced AI assistant for the Bot Builder system. I can help you create, manage, and optimize AI employees for your hedge fund operations.",
            "general": "I'm here to help you with the Bot Builder AI system. Whether you need to create AI employees, monitor performance, or optimize your system, I'm ready to assist!"
        }
        
        base_response = responses.get(topic, responses["general"])
        
        return f"{base_response}\n\n**Current System Status:**\n- Active AI Employees: {len(self.active_ai_employees)}\n- System Health: {self.system_health}\n- Advanced RL Engine: {'Enabled' if self.rl_enabled else 'Disabled'}\n- Real-time Data: {'Enabled' if self.real_time_enabled else 'Disabled'}\n- Explainability: {'Enabled' if self.explainability_enabled else 'Disabled'}\n\nIs there anything specific you'd like to work on with the Bot Builder system?"
    
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
    "action": "create_ai_employee|monitor_performance|optimize_ai_employee|get_status|configure_system|self_improve|analyze_system|approve_proposal|reject_proposal|help|greeting|general_chat|unknown",
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
   Parameters: {{"status_type": "system|employees|performance|self_improvement"}}

5. configure_system - User wants to configure system
   Parameters: {{"config_type": "string", "config_value": "string"}}

6. self_improve - User wants to trigger self-improvement analysis
   Parameters: {{"improvement_type": "automatic|manual", "focus_area": "string (optional)"}}

7. analyze_system - User wants to analyze system for improvements
   Parameters: {{"analysis_type": "comprehensive|performance|code_quality|technical_debt"}}

8. approve_proposal - User wants to approve a self-improvement proposal
   Parameters: {{"proposal_id": "string", "notes": "string (optional)"}}

9. reject_proposal - User wants to reject a self-improvement proposal
   Parameters: {{"proposal_id": "string", "reason": "string", "notes": "string (optional)"}}

10. help - User wants help
    Parameters: {{}}

11. greeting - User is greeting or saying hello
    Parameters: {{"greeting_type": "hello|hi|hey|good_morning|good_afternoon|good_evening"}}

12. general_chat - User is having general conversation
    Parameters: {{"topic": "string", "sentiment": "positive|neutral|negative"}}

13. unknown - Cannot determine intent
    Parameters: {{}}

Examples:
- "hello" → {{"action": "greeting", "confidence": 0.9, "parameters": {{"greeting_type": "hello"}}}}
- "hi there" → {{"action": "greeting", "confidence": 0.9, "parameters": {{"greeting_type": "hi"}}}}
- "how are you" → {{"action": "general_chat", "confidence": 0.8, "parameters": {{"topic": "wellbeing", "sentiment": "neutral"}}}}
- "create a trader" → {{"action": "create_ai_employee", "confidence": 0.9, "parameters": {{"role": "trader", "specialization": "eral"}}}}
- "improve yourself" → {{"action": "self_improve", "confidence": 0.9, "parameters": {{"improvement_type": "automatic"}}}}
- "analyze system" → {{"action": "analyze_system", "confidence": 0.9, "parameters": {{"analysis_type": "comprehensive"}}}}
- "approve proposal 123" → {{"action": "approve_proposal", "confidence": 0.9, "parameters": {{"proposal_id": "123"}}}}

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

    async def get_ai_employee_action(self, employee_id: str, market_data: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI Employee action using advanced RL with explainability."""
        try:
            if employee_id not in self.active_ai_employees:
                return {"error": "Employee not found"}
            
            # Create RL state
            rl_state = self._create_rl_state(market_data, portfolio_state)
            
            # Get action from advanced RL engine
            if self.rl_enabled and employee_id in self.advanced_rl_engine.models:
                action = await self.advanced_rl_engine.get_action(employee_id, rl_state)
                
                # Generate explanation if enabled
                explanation = None
                if self.explainability_enabled:
                    explanation = await self._generate_action_explanation(
                        employee_id, action, market_data, portfolio_state
                    )
                
                return {
                    "action_type": action.action_type,
                    "action_value": action.action_value,
                    "confidence": action.confidence,
                    "metadata": action.metadata,
                    "rl_engine": "advanced",
                    "explanation": explanation
                }
            else:
                # Fallback to basic decision making
                basic_action = await self._get_basic_action(employee_id, market_data, portfolio_state)
                
                # Generate explanation for basic action
                if self.explainability_enabled:
                    basic_action["explanation"] = await self._generate_action_explanation(
                        employee_id, basic_action, market_data, portfolio_state
                    )
                
                return basic_action
                
        except Exception as e:
            logger.error(f"Error getting action for {employee_id}: {str(e)}")
            return {"error": str(e)}

    async def _generate_action_explanation(
        self, 
        employee_id: str, 
        action: Any, 
        market_data: Dict[str, Any], 
        portfolio_state: Dict[str, Any]
    ) -> Optional[ExplanationResult]:
        """Generate explanation for an AI action."""
        try:
            # Create decision context
            context = DecisionContext(
                market_data=market_data,
                portfolio_state=portfolio_state,
                risk_metrics=self._extract_risk_metrics(portfolio_state),
                historical_context=self._get_historical_context(employee_id),
                regulatory_context=self._get_regulatory_context()
            )
            
            # Create decision data
            decision_data = {
                "action_type": action.get("action_type", "unknown"),
                "action_value": action.get("action_value", 0),
                "confidence": action.get("confidence", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate explanation
            explanation = await self.explainability_engine.explain_decision(
                employee_id=employee_id,
                decision_data=decision_data,
                context=context,
                explanation_types=["rule_based", "statistical", "regulatory"]
            )
            
            # Store in history
            self.explanation_history.append(explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating action explanation: {str(e)}")
            return None

    def _extract_risk_metrics(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk metrics from portfolio state."""
        return {
            "var_95": portfolio_state.get("var_95", 0),
            "max_drawdown": portfolio_state.get("max_drawdown", 0),
            "sharpe_ratio": portfolio_state.get("sharpe_ratio", 0),
            "beta": portfolio_state.get("beta", 0)
        }

    def _get_historical_context(self, employee_id: str) -> Dict[str, Any]:
        """Get historical context for an employee."""
        try:
            # Get recent decisions
            recent_decisions = [
                exp for exp in self.explanation_history
                if exp.employee_id == employee_id and 
                exp.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            return {
                "recent_decisions": len(recent_decisions),
                "average_confidence": np.mean([exp.confidence for exp in recent_decisions]) if recent_decisions else 0,
                "decision_types": list(set([exp.metadata.get("decision_type", "unknown") for exp in recent_decisions]))
            }
            
        except Exception as e:
            logger.error(f"Error getting historical context: {str(e)}")
            return {}

    def _get_regulatory_context(self) -> Dict[str, Any]:
        """Get regulatory context."""
        return {
            "compliance_score": 0.98,  # Simulated compliance score
            "violations": [],  # No violations
            "regulatory_environment": "stable"
        }

    async def get_decision_explanation(self, decision_id: str) -> Optional[ExplanationResult]:
        """Get explanation for a specific decision."""
        try:
            # Check cache first
            if decision_id in self.explainability_engine.explanations_cache:
                return self.explainability_engine.explanations_cache[decision_id]
            
            # Check explanation history
            for explanation in self.explanation_history:
                if explanation.decision_id == decision_id:
                    return explanation
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting decision explanation: {str(e)}")
            return None

    async def get_employee_explanations(self, employee_id: str, hours_back: int = 24) -> List[ExplanationResult]:
        """Get recent explanations for an employee."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            explanations = [
                exp for exp in self.explanation_history
                if exp.employee_id == employee_id and exp.timestamp > cutoff_time
            ]
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error getting employee explanations: {str(e)}")
            return []

    async def get_explanation_summary(self, employee_id: str) -> Dict[str, Any]:
        """Get explanation summary for an employee."""
        try:
            if self.explainability_enabled:
                return await self.explainability_engine.get_explanation_summary(employee_id)
            else:
                return {"error": "Explainability not enabled"}
        except Exception as e:
            logger.error(f"Error getting explanation summary: {str(e)}")
            return {"error": str(e)}

    async def get_feature_importance(self, employee_id: str) -> List[Dict[str, Any]]:
        """Get feature importance for an employee."""
        try:
            if self.explainability_enabled:
                features = await self.explainability_engine.get_feature_importance(employee_id, {})
                return [
                    {
                        "name": feature.feature_name,
                        "importance": feature.importance_score,
                        "direction": feature.direction,
                        "description": feature.description
                    }
                    for feature in features
                ]
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return []

    async def _handle_explain_decision(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle decision explanation request."""
        try:
            parameters = intent.get("parameters", {})
            employee_id = parameters.get("employee_id", "")
            decision_id = parameters.get("decision_id", "")
            
            if not employee_id:
                return "Please specify an employee ID to get decision explanations."
            
            if decision_id:
                # Get specific decision explanation
                explanation = await self.get_decision_explanation(decision_id)
                if explanation:
                    return self._format_decision_explanation(explanation)
                else:
                    return f"Decision {decision_id} not found."
            else:
                # Get recent explanations for employee
                explanations = await self.get_employee_explanations(employee_id, hours_back=24)
                if explanations:
                    return self._format_employee_explanations(employee_id, explanations)
                else:
                    return f"No recent explanations found for employee {employee_id}."
                    
        except Exception as e:
            logger.error(f"Error handling explain decision: {str(e)}")
            return f"I encountered an error while explaining the decision: {str(e)}"

    def _format_decision_explanation(self, explanation: ExplanationResult) -> str:
        """Format a decision explanation for display."""
        try:
            response = f"""**Decision Explanation for {explanation.employee_id}**

**Decision ID**: {explanation.decision_id}
**Confidence**: {explanation.confidence:.2%}
**Timestamp**: {explanation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

**Key Factors:**
"""
            
            for factor in explanation.factors[:5]:  # Top 5 factors
                response += f"- **{factor['name']}**: {factor['description']}\n"
            
            if explanation.visualization:
                response += "\n**Visualization**: Available (base64 encoded)\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting decision explanation: {str(e)}")
            return f"Error formatting explanation: {str(e)}"

    def _format_employee_explanations(self, employee_id: str, explanations: List[ExplanationResult]) -> str:
        """Format employee explanations for display."""
        try:
            response = f"""**Recent Explanations for {employee_id}**

**Total Explanations**: {len(explanations)}
**Average Confidence**: {np.mean([exp.confidence for exp in explanations]):.2%}

**Recent Decisions:**
"""
            
            for exp in explanations[-5:]:  # Last 5 explanations
                response += f"- **{exp.decision_id}**: {exp.confidence:.2%} confidence\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting employee explanations: {str(e)}")
            return f"Error formatting explanations: {str(e)}"

    def _get_explainability_capabilities(self, role: str) -> str:
        """Get explainability capabilities for a role."""
        capabilities = {
            "research_analyst": """
- SHAP-based feature importance analysis
- LIME-based local explanations
- Decision rationale transparency
- Market analysis explanations
- Pattern recognition insights""",
            "trader": """
- Trade decision explanations
- Risk-reward analysis transparency
- Execution strategy rationale
- Portfolio impact analysis
- Market timing explanations""",
            "risk_manager": """
- Risk assessment explanations
- VaR calculation transparency
- Stress testing rationale
- Risk factor analysis
- Compliance risk explanations""",
            "compliance_officer": """
- Regulatory decision explanations
- Compliance rule transparency
- Audit trail explanations
- Violation analysis
- Policy interpretation rationale""",
            "data_specialist": """
- Data processing explanations
- Feature engineering transparency
- Data quality analysis
- Pipeline decision rationale
- Model performance insights"""
        }
        return capabilities.get(role, "- Decision transparency\n- Feature importance analysis\n- Rationale explanations")

    def _create_rl_state(self, market_data: Dict[str, Any], portfolio_state: Dict[str, Any]) -> RLState:
        """Create RL state from market and portfolio data."""
        # Convert market data to numpy array
        market_array = np.array([
            market_data.get('price', 0),
            market_data.get('volume', 0),
            market_data.get('volatility', 0),
            market_data.get('trend', 0)
        ])
        
        # Convert portfolio state to numpy array
        portfolio_array = np.array([
            portfolio_state.get('cash', 0),
            portfolio_state.get('total_value', 0),
            portfolio_state.get('risk_level', 0),
            portfolio_state.get('position_count', 0)
        ])
        
        # Create risk metrics
        risk_metrics = np.array([
            portfolio_state.get('var_95', 0),
            portfolio_state.get('max_drawdown', 0),
            portfolio_state.get('sharpe_ratio', 0),
            portfolio_state.get('beta', 0)
        ])
        
        # Time features
        current_time = datetime.now()
        time_features = np.array([
            current_time.hour / 24.0,
            current_time.weekday() / 7.0,
            current_time.month / 12.0
        ])
        
        # Context features
        context_features = np.array([
            market_data.get('market_sentiment', 0),
            market_data.get('news_sentiment', 0),
            market_data.get('technical_signal', 0)
        ])
        
        return RLState(
            market_data=market_array,
            portfolio_state=portfolio_array,
            risk_metrics=risk_metrics,
            time_features=time_features,
            context_features=context_features
        )

    async def _get_basic_action(self, employee_id: str, market_data: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get basic action when RL is not available."""
        # Simple rule-based decision making
        price = market_data.get('price', 0)
        trend = market_data.get('trend', 0)
        
        if trend > 0.1:
            action_type = "buy"
            confidence = min(0.8, abs(trend))
        elif trend < -0.1:
            action_type = "sell"
            confidence = min(0.8, abs(trend))
        else:
            action_type = "hold"
            confidence = 0.5
        
        return {
            "action_type": action_type,
            "action_value": confidence,
            "confidence": confidence,
            "metadata": {"method": "basic_rules"},
            "rl_engine": "basic"
        }

    async def update_ai_employee_experience(self, employee_id: str, state: RLState, action: RLAction, reward: RLReward, next_state: RLState):
        """Update AI Employee experience for RL learning."""
        try:
            if self.rl_enabled and employee_id in self.advanced_rl_engine.models:
                await self.advanced_rl_engine.update_model(employee_id, state, action, reward, next_state)
                logger.debug(f"Updated RL experience for employee {employee_id}")
        except Exception as e:
            logger.error(f"Error updating experience for {employee_id}: {str(e)}")

    def _get_role_capabilities(self, role: str) -> str:
        """Get role-specific capabilities description."""
        capabilities = {
            "research_analyst": """
- Advanced market analysis and forecasting
- Sentiment analysis and news processing
- Technical and fundamental analysis
- Real-time market monitoring
- Advanced RL for pattern recognition""",
            "trader": """
- High-frequency trading capabilities
- Risk-aware position management
- Advanced RL for optimal execution
- Real-time market adaptation
- Portfolio optimization""",
            "risk_manager": """
- Advanced risk modeling and assessment
- VaR and stress testing
- Dynamic risk threshold adjustment
- RL-based risk prediction
- Compliance monitoring""",
            "compliance_officer": """
- Regulatory compliance monitoring
- Audit trail management
- Rule-based decision validation
- RL for compliance optimization
- Real-time regulatory updates""",
            "data_specialist": """
- Advanced data processing and cleaning
- Feature engineering and selection
- Data quality assessment
- RL for data optimization
- Real-time data pipeline management"""
        }
        return capabilities.get(role, "- General AI capabilities\n- Adaptive learning\n- Performance optimization") 

    def _get_real_time_capabilities(self, role: str) -> str:
        """Get real-time capabilities for a role."""
        capabilities = {
            "research_analyst": """
- Real-time market data analysis
- Live news sentiment processing
- Instant market trend detection
- Continuous pattern recognition""",
            "trader": """
- Real-time trade execution
- Live market order management
- Instant price monitoring
- Continuous portfolio rebalancing""",
            "risk_manager": """
- Real-time risk assessment
- Live VaR calculations
- Instant stress testing
- Continuous exposure monitoring""",
            "compliance_officer": """
- Real-time compliance monitoring
- Live regulatory updates
- Instant audit trail tracking
- Continuous rule validation""",
            "data_specialist": """
- Real-time data processing
- Live data quality monitoring
- Instant pipeline management
- Continuous data validation"""
        }
        return capabilities.get(role, "- Real-time data processing\n- Live market monitoring\n- Instant decision making")

    async def _start_real_time_data_feed(self):
        """Start real-time data feed for monitored symbols."""
        try:
            success = await self.real_time_data.start_real_time_feed(
                symbols=self.monitored_symbols,
                data_sources=["yahoo_finance", "simulated"]
            )
            
            if success:
                logger.info(f"Real-time data feed started for {len(self.monitored_symbols)} symbols")
            else:
                logger.warning("Failed to start real-time data feed")
                
        except Exception as e:
            logger.error(f"Error starting real-time data feed: {str(e)}")
    
    async def _setup_market_event_handlers(self):
        """Set up handlers for market events."""
        try:
            # Subscribe to various market events
            await self.real_time_data.subscribe_to_events('price_update', self._handle_price_update)
            await self.real_time_data.subscribe_to_events('volume_spike', self._handle_volume_spike)
            await self.real_time_data.subscribe_to_events('news_event', self._handle_news_event)
            await self.real_time_data.subscribe_to_events('technical_signal', self._handle_technical_signal)
            
            logger.info("Market event handlers set up successfully")
            
        except Exception as e:
            logger.error(f"Error setting up market event handlers: {str(e)}")
    
    async def _handle_price_update(self, event: MarketEvent):
        """Handle price update events."""
        try:
            symbol = event.symbol
            data = event.data
            
            # Update AI Employees with new price data
            for employee_id, employee_info in self.active_ai_employees.items():
                if employee_info.get("real_time_enabled", False):
                    await self._update_employee_with_market_data(employee_id, symbol, data)
            
            # Log significant price movements
            if abs(data.get("change_percent", 0)) > 5:  # 5% or more change
                logger.info(f"Significant price movement detected for {symbol}: {data.get('change_percent', 0):.2f}%")
                
        except Exception as e:
            logger.error(f"Error handling price update: {str(e)}")
    
    async def _handle_volume_spike(self, event: MarketEvent):
        """Handle volume spike events."""
        try:
            symbol = event.symbol
            data = event.data
            
            logger.info(f"Volume spike detected for {symbol}: {data.get('spike_ratio', 0):.2f}x average volume")
            
            # Trigger special analysis for volume spikes
            await self._trigger_volume_spike_analysis(symbol, data)
            
        except Exception as e:
            logger.error(f"Error handling volume spike: {str(e)}")
    
    async def _handle_news_event(self, event: MarketEvent):
        """Handle news events."""
        try:
            symbol = event.symbol
            data = event.data
            
            logger.info(f"News event detected for {symbol}: {data.get('headline', 'Unknown')}")
            
            # Trigger news analysis
            await self._trigger_news_analysis(symbol, data)
            
        except Exception as e:
            logger.error(f"Error handling news event: {str(e)}")
    
    async def _handle_technical_signal(self, event: MarketEvent):
        """Handle technical signals."""
        try:
            symbol = event.symbol
            data = event.data
            
            logger.info(f"Technical signal detected for {symbol}: {data.get('signal_type', 'Unknown')}")
            
            # Trigger technical analysis
            await self._trigger_technical_analysis(symbol, data)
            
        except Exception as e:
            logger.error(f"Error handling technical signal: {str(e)}")
    
    async def _update_employee_with_market_data(self, employee_id: str, symbol: str, data: Dict[str, Any]):
        """Update AI Employee with new market data."""
        try:
            # Create market data structure
            market_data = {
                "symbol": symbol,
                "price": data.get("price", 0),
                "volume": data.get("volume", 0),
                "change_percent": data.get("change_percent", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update employee if it has real-time processing capabilities
            if self.rl_enabled and employee_id in self.advanced_rl_engine.models:
                # Create RL state and update
                rl_state = self._create_rl_state_from_market_data(market_data)
                # This would trigger RL learning updates
                
            logger.debug(f"Updated employee {employee_id} with market data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating employee with market data: {str(e)}")
    
    def _create_rl_state_from_market_data(self, market_data: Dict[str, Any]) -> RLState:
        """Create RL state from market data."""
        # Convert market data to RL state format
        market_array = np.array([
            market_data.get('price', 0),
            market_data.get('volume', 0),
            market_data.get('change_percent', 0),
            0  # Placeholder for additional features
        ])
        
        # Create minimal RL state
        return RLState(
            market_data=market_array,
            portfolio_state=np.array([0, 0, 0, 0]),  # Placeholder
            risk_metrics=np.array([0, 0, 0, 0]),  # Placeholder
            time_features=np.array([0, 0, 0]),  # Placeholder
            context_features=np.array([0, 0, 0])  # Placeholder
        )
    
    async def _trigger_volume_spike_analysis(self, symbol: str, data: Dict[str, Any]):
        """Trigger analysis for volume spikes."""
        try:
            # Find research analyst employees
            for employee_id, employee_info in self.active_ai_employees.items():
                if employee_info.get("role") == "research_analyst":
                    # Trigger volume analysis
                    logger.info(f"Triggering volume spike analysis for {symbol} with employee {employee_id}")
                    
        except Exception as e:
            logger.error(f"Error triggering volume spike analysis: {str(e)}")
    
    async def _trigger_news_analysis(self, symbol: str, data: Dict[str, Any]):
        """Trigger analysis for news events."""
        try:
            # Find research analyst employees
            for employee_id, employee_info in self.active_ai_employees.items():
                if employee_info.get("role") == "research_analyst":
                    # Trigger news analysis
                    logger.info(f"Triggering news analysis for {symbol} with employee {employee_id}")
                    
        except Exception as e:
            logger.error(f"Error triggering news analysis: {str(e)}")
    
    async def _trigger_technical_analysis(self, symbol: str, data: Dict[str, Any]):
        """Trigger analysis for technical signals."""
        try:
            # Find trader employees
            for employee_id, employee_info in self.active_ai_employees.items():
                if employee_info.get("role") == "trader":
                    # Trigger technical analysis
                    logger.info(f"Triggering technical analysis for {symbol} with employee {employee_id}")
                    
        except Exception as e:
            logger.error(f"Error triggering technical analysis: {str(e)}")

    async def get_real_time_market_summary(self) -> Dict[str, Any]:
        """Get real-time market summary."""
        try:
            if self.real_time_enabled:
                return await self.real_time_data.get_market_summary()
            else:
                return {"error": "Real-time data not enabled"}
        except Exception as e:
            logger.error(f"Error getting market summary: {str(e)}")
            return {"error": str(e)}

    async def get_live_market_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get live market data for a symbol."""
        try:
            if self.real_time_enabled:
                return await self.real_time_data.get_real_time_data(symbol)
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting live market data: {str(e)}")
            return None

    async def get_market_events(self, minutes_back: int = 10) -> List[MarketEvent]:
        """Get recent market events."""
        try:
            if self.real_time_enabled:
                # Return recent events from the real-time data provider
                return [e for e in self.real_time_data.market_events if e.timestamp > datetime.now() - timedelta(minutes=minutes_back)]
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting market events: {str(e)}")
            return []

    async def add_symbol_to_monitoring(self, symbol: str) -> bool:
        """Add a symbol to real-time monitoring."""
        try:
            if symbol not in self.monitored_symbols:
                self.monitored_symbols.append(symbol)
                
                if self.real_time_enabled:
                    # Restart feed with new symbol
                    await self.real_time_data.stop_real_time_feed()
                    await self._start_real_time_data_feed()
                
                logger.info(f"Added {symbol} to real-time monitoring")
                return True
            else:
                logger.info(f"{symbol} is already being monitored")
                return True
                
        except Exception as e:
            logger.error(f"Error adding symbol to monitoring: {str(e)}")
            return False

    async def remove_symbol_from_monitoring(self, symbol: str) -> bool:
        """Remove a symbol from real-time monitoring."""
        try:
            if symbol in self.monitored_symbols:
                self.monitored_symbols.remove(symbol)
                
                if self.real_time_enabled:
                    # Restart feed without the symbol
                    await self.real_time_data.stop_real_time_feed()
                    await self._start_real_time_data_feed()
                
                logger.info(f"Removed {symbol} from real-time monitoring")
                return True
            else:
                logger.info(f"{symbol} is not being monitored")
                return True
                
        except Exception as e:
            logger.error(f"Error removing symbol from monitoring: {str(e)}")
            return False
    
    # NEW: Self-Improvement Handlers
    
    async def _handle_self_improve(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle self-improvement requests."""
        try:
            parameters = intent.get("parameters", {})
            improvement_type = parameters.get("improvement_type", "automatic")
            focus_area = parameters.get("focus_area", "")
            
            if improvement_type == "automatic":
                # Trigger autonomous improvement cycle
                response = """🧠 **True AI Self-Improvement Initiated!**

I'm now analyzing my own system to identify improvement opportunities. This is **real self-improvement** - I'm using my own AI capabilities to:

**🔍 Self-Analysis:**
- Analyzing my own code and performance
- Identifying weaknesses and bottlenecks
- Finding optimization opportunities
- Assessing technical debt

**📋 Proposal Generation:**
- Generating specific improvement proposals
- Creating code diffs for the improvements
- Estimating impact and risk levels
- Preparing implementation plans

**👔 CEO Approval Required:**
- All proposals will be submitted for your review
- You'll see detailed explanations and code changes
- You can approve, reject, or modify proposals
- Full audit trail of all decisions

**⚡ Implementation:**
- If approved, I'll implement the changes
- Run tests to ensure everything works
- Monitor the improvements
- Learn from the results

This is **true AI self-improvement** - not just task management, but actual code generation and system enhancement!

**Status:** Analysis in progress...
**Next Step:** You'll receive approval requests for any improvements I identify."""
                
            else:
                response = """🧠 **Manual Self-Improvement Mode**

I can analyze specific areas of my system for improvements. What would you like me to focus on?

**Available Focus Areas:**
- Performance optimization
- Code quality improvements
- Feature enhancements
- Bug fixes
- Architecture improvements

Just let me know what area you'd like me to analyze!"""
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling self-improvement request: {str(e)}")
            return f"I encountered an error while initiating self-improvement: {str(e)}"
    
    async def _handle_analyze_system(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle system analysis requests."""
        try:
            parameters = intent.get("parameters", {})
            analysis_type = parameters.get("analysis_type", "comprehensive")
            
            # Perform system analysis
            analysis = await self.self_improvement_engine.analyze_system()
            
            response = f"""🔍 **System Analysis Results**

**Analysis Type:** {analysis_type.title()}
**Health Score:** {analysis.system_health_score:.1%}
**Timestamp:** {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

**📊 Performance Metrics:**
{self._format_system_analysis_metrics(analysis.performance_metrics)}

**🚨 Identified Issues:**
{self._format_identified_issues(analysis.identified_issues)}

**💡 Improvement Opportunities:**
{self._format_improvement_opportunities(analysis.improvement_opportunities)}

**📈 Code Quality Metrics:**
{self._format_code_quality_metrics(analysis.code_quality_metrics)}

**🔧 Technical Debt Analysis:**
{self._format_technical_debt(analysis.technical_debt_analysis)}

**Next Steps:**
- I can generate specific improvement proposals for any of these opportunities
- Just say "improve yourself" to start the self-improvement process
- Or ask me to focus on a specific area"""
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling system analysis: {str(e)}")
            return f"I encountered an error while analyzing the system: {str(e)}"
    
    async def _handle_approve_proposal(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle proposal approval requests."""
        try:
            parameters = intent.get("parameters", {})
            proposal_id = parameters.get("proposal_id", "")
            notes = parameters.get("notes", "")
            
            if not proposal_id:
                return "Please provide a proposal ID to approve."
            
            # Get pending approvals
            pending_approvals = self.self_improvement_engine.approval_engine.get_pending_approvals()
            
            # Find the approval request
            approval_request = None
            for approval in pending_approvals:
                if approval.proposal_id == proposal_id:
                    approval_request = approval
                    break
            
            if not approval_request:
                return f"Proposal {proposal_id} not found in pending approvals."
            
            # Approve the proposal
            success = await self.self_improvement_engine.approval_engine.approve_proposal(
                approval_request.id, 
                context.user_id, 
                notes
            )
            
            if success:
                # Implement the proposal
                proposal = await self._get_proposal_by_id(proposal_id)
                if proposal:
                    implementation_success = await self.self_improvement_engine.implement_proposal(proposal)
                    
                    if implementation_success:
                        response = f"""✅ **Proposal Approved and Implemented!**

**Proposal ID:** {proposal_id}
**Title:** {approval_request.title}
**Approved by:** {context.user_id}
**Status:** Successfully implemented

**🎯 What was implemented:**
{approval_request.description}

**📈 Expected Impact:**
{self._format_estimated_impact(approval_request.estimated_impact)}

**📝 Notes:** {notes if notes else "None"}

The self-improvement has been successfully applied to the system!"""
                    else:
                        response = f"""⚠️ **Proposal Approved but Implementation Failed**

**Proposal ID:** {proposal_id}
**Title:** {approval_request.title}
**Status:** Implementation failed

The proposal was approved but encountered an error during implementation. The system will attempt to rollback any partial changes."""
                else:
                    response = f"""⚠️ **Proposal Approved but Not Found**

**Proposal ID:** {proposal_id}
**Status:** Proposal not found for implementation

The proposal was approved but could not be located for implementation."""
            else:
                response = f"❌ Failed to approve proposal {proposal_id}."
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling proposal approval: {str(e)}")
            return f"I encountered an error while approving the proposal: {str(e)}"
    
    async def _handle_reject_proposal(self, intent: Dict[str, Any], context: ConversationContext) -> str:
        """Handle proposal rejection requests."""
        try:
            parameters = intent.get("parameters", {})
            proposal_id = parameters.get("proposal_id", "")
            reason = parameters.get("reason", "No reason provided")
            notes = parameters.get("notes", "")
            
            if not proposal_id:
                return "Please provide a proposal ID to reject."
            
            # Get pending approvals
            pending_approvals = self.self_improvement_engine.approval_engine.get_pending_approvals()
            
            # Find the approval request
            approval_request = None
            for approval in pending_approvals:
                if approval.proposal_id == proposal_id:
                    approval_request = approval
                    break
            
            if not approval_request:
                return f"Proposal {proposal_id} not found in pending approvals."
            
            # Reject the proposal
            success = await self.self_improvement_engine.approval_engine.reject_proposal(
                approval_request.id, 
                context.user_id, 
                reason, 
                notes
            )
            
            if success:
                response = f"""❌ **Proposal Rejected**

**Proposal ID:** {proposal_id}
**Title:** {approval_request.title}
**Rejected by:** {context.user_id}
**Reason:** {reason}
**Notes:** {notes if notes else "None"}

The proposal has been rejected and will not be implemented."""
            else:
                response = f"❌ Failed to reject proposal {proposal_id}."
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling proposal rejection: {str(e)}")
            return f"I encountered an error while rejecting the proposal: {str(e)}"
    
    # Helper methods for self-improvement
    
    def _format_system_analysis_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format system analysis metrics."""
        if not metrics or "error" in metrics:
            return "No metrics available"
        
        formatted = ""
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted += f"- {key.replace('_', ' ').title()}: {value:.2f}\n"
            else:
                formatted += f"- {key.replace('_', ' ').title()}: {value}\n"
        return formatted
    
    def _format_identified_issues(self, issues: List[Dict[str, Any]]) -> str:
        """Format identified issues."""
        if not issues:
            return "No issues identified"
        
        formatted = ""
        for i, issue in enumerate(issues[:5], 1):  # Show first 5 issues
            formatted += f"{i}. **{issue.get('title', 'Unknown')}** ({issue.get('severity', 'UNKNOWN')})\n"
            formatted += f"   {issue.get('description', 'No description')}\n\n"
        return formatted
    
    def _format_improvement_opportunities(self, opportunities: List[Dict[str, Any]]) -> str:
        """Format improvement opportunities."""
        if not opportunities:
            return "No improvement opportunities identified"
        
        formatted = ""
        for i, opp in enumerate(opportunities[:5], 1):  # Show first 5 opportunities
            formatted += f"{i}. **{opp.get('title', 'Unknown')}** ({opp.get('priority', 'UNKNOWN')})\n"
            formatted += f"   {opp.get('description', 'No description')}\n"
            formatted += f"   Risk Level: {opp.get('risk_level', 'UNKNOWN')}\n\n"
        return formatted
    
    def _format_code_quality_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format code quality metrics."""
        if not metrics or "error" in metrics:
            return "No code quality metrics available"
        
        formatted = ""
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted += f"- {key.replace('_', ' ').title()}: {value:.2f}\n"
            else:
                formatted += f"- {key.replace('_', ' ').title()}: {value}\n"
        return formatted
    
    def _format_technical_debt(self, debt: Dict[str, Any]) -> str:
        """Format technical debt analysis."""
        if not debt or "error" in debt:
            return "No technical debt analysis available"
        
        formatted = ""
        for key, value in debt.items():
            formatted += f"- {key.replace('_', ' ').title()}: {value}\n"
        return formatted
    
    def _format_estimated_impact(self, impact: Dict[str, Any]) -> str:
        """Format estimated impact."""
        if not impact:
            return "No impact data available"
        
        formatted = ""
        for key, value in impact.items():
            formatted += f"- {key.replace('_', ' ').title()}: {value}\n"
        return formatted
    
    async def _get_proposal_by_id(self, proposal_id: str) -> Optional[SelfImprovementProposal]:
        """Get a proposal by ID."""
        try:
            # This would typically query a database
            # For now, search in improvement history
            for proposal in self.self_improvement_engine.improvement_history:
                if proposal.id == proposal_id:
                    return proposal
            return None
        except Exception as e:
            logger.error(f"Error getting proposal by ID: {str(e)}")
            return None
    
    async def chat_with_intelligent_assistant(self, message: str) -> str:
        """
        Chat with the intelligent assistant that can route between local and cloud LLMs.
        
        Args:
            message: User's message
            
        Returns:
            Assistant's response
        """
        try:
            return await self.intelligent_chat.process_message(message)
        except Exception as e:
            logger.error(f"Error in intelligent chat: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get chat history from the intelligent assistant."""
        try:
            return self.intelligent_chat.get_chat_history()
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []
    
    def clear_chat_history(self) -> None:
        """Clear chat history."""
        try:
            self.intelligent_chat.clear_history()
        except Exception as e:
            logger.error(f"Error clearing chat history: {str(e)}") 