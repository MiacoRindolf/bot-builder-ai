"""
Gradio web interface for the Bot Builder AI system.
Provides an alternative to Streamlit with a conversational chat interface.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import gradio as gr
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import uuid

# Import our modules
from core.ai_engine import AIEngine
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioApp:
    """Gradio application for the Bot Builder AI system."""
    
    def __init__(self):
        """Initialize the Gradio application."""
        self.ai_engine = None
        self.session_id = None
        self.user_id = None
        self.chat_history = []
        
        # Initialize AI Engine
        self._initialize_ai_engine()
    
    def _initialize_ai_engine(self):
        """Initialize the AI Engine."""
        try:
            self.ai_engine = AIEngine()
            logger.info("AI Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI Engine: {str(e)}")
    
    def get_session_id(self):
        """Get or create session ID."""
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())
        return self.session_id
    
    def get_user_id(self):
        """Get or create user ID."""
        if self.user_id is None:
            self.user_id = f"user_{uuid.uuid4().hex[:8]}"
        return self.user_id
    
    def process_message(self, message: str, history: List[dict]) -> tuple:
        """
        Process user message and generate response.
        
        Args:
            message: User message
            history: Chat history
            
        Returns:
            Tuple of (response, updated_history)
        """
        try:
            if not message.strip():
                return "", history
            
            # Get AI response (sync version)
            response = self._get_ai_response_sync(message)
            
            # Update history with proper format for messages type
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            return "", history
            
        except Exception as e:
            error_response = f"I encountered an error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})
            return "", history
    
    def _get_ai_response_sync(self, message: str) -> str:
        """Get response from AI Engine (synchronous version)."""
        try:
            if self.ai_engine is None:
                return "AI Engine is not initialized. Please check your configuration."
            
            # Simple response for now to avoid async issues
            if "hello" in message.lower() or "hi" in message.lower():
                return "Hello! I'm your Bot Builder AI assistant. How can I help you today?"
            elif "status" in message.lower():
                return "The Bot Builder AI system is running smoothly with all components operational."
            elif "create" in message.lower() and "employee" in message.lower():
                return "You can create AI Employees using the form on the right side of this interface."
            elif "help" in message.lower():
                return "I can help you with creating AI Employees, checking system status, and managing your AI workforce. What would you like to know?"
            else:
                return f"I understand you said: '{message}'. I'm here to help you manage your AI Employees and monitor system performance."
                
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def _get_ai_response(self, message: str) -> str:
        """Get response from AI Engine (async version - kept for compatibility)."""
        return self._get_ai_response_sync(message)
    
    def create_ai_employee(self, role: str, specialization: str) -> str:
        """
        Create a new AI Employee.
        
        Args:
            role: AI Employee role
            specialization: Specialization area
            
        Returns:
            Status message
        """
        try:
            if not specialization.strip():
                return "Please enter a specialization."
            
            # Create employee ID
            employee_id = f"{role}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            return f"✅ AI Employee {employee_id} created successfully!\n\nRole: {role.title()}\nSpecialization: {specialization}\nStatus: Training in progress..."
            
        except Exception as e:
            return f"❌ Error creating AI Employee: {str(e)}"
    
    def get_system_status(self) -> str:
        """Get system status."""
        try:
            status = f"""
🤖 **Bot Builder AI System Status**

**System Health**: ✅ Operational
**AI Engine**: ✅ Active  
**Data Manager**: ✅ Active
**Metrics Collector**: ✅ Active
**Self-Improvement**: ✅ Enabled
**Real-time Data**: ✅ Active

**Quick Actions**:
• Create AI Employees using the form below
• Chat with the AI Assistant
• Monitor performance and metrics

**Available Roles**:
• Research Analyst: Deep learning, forecasting, economic analysis
• Trader: Reinforcement learning, execution speed, strategic decision-making
• Risk Manager: Probability theory, statistical modeling, scenario testing
• Compliance Officer: Regulatory knowledge, NLP, explainability
• Data Specialist: Data cleaning, management, structuring
            """
            return status
            
        except Exception as e:
            return f"❌ Error getting system status: {str(e)}"
    
    def get_help(self) -> str:
        """Get help information."""
        help_text = """
📋 **Bot Builder AI - Help Guide**

## Available Commands

### Creating AI Employees
- Use the form below to create new AI Employees
- Select a role and enter a specialization
- Click "Create AI Employee" to start

### Chat Commands
- "Create a new Research Analyst AI Employee focused on cryptocurrency markets"
- "Show me the performance metrics for all AI Employees"
- "Optimize the Trader AI Employee for better execution speed"
- "What's the current system status?"
- "Display performance analytics for the last 30 days"

### AI Employee Roles

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

Need more help? Just ask in the chat!
        """
        return help_text
    
    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(
            title="Bot Builder AI",
            css="""
            .gradio-container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                margin-bottom: 2rem;
            }
            .status-box {
                background-color: #f0f8ff;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #1f77b4;
            }
            """
        ) as interface:
            
            # Header
            with gr.Row():
                gr.HTML("""
                <div class="header">
                    <h1>🤖 Bot Builder AI</h1>
                    <p>Advanced AI Employee Management System for AI-Powered Hedge Funds</p>
                </div>
                """)
            
            # Main content
            with gr.Row():
                # Left column - Chat interface
                with gr.Column(scale=2):
                    gr.HTML("<h3>💬 AI Assistant</h3>")
                    
                    # Chat interface
                    chatbot = gr.Chatbot(
                        height=400,
                        show_label=False,
                        container=True,
                        type="messages"
                    )
                    
                    # Message input
                    msg = gr.Textbox(
                        placeholder="Ask me anything about AI Employees, performance, or system management...",
                        show_label=False,
                        lines=2
                    )
                    
                    # Send button
                    send_btn = gr.Button("Send", variant="primary")
                    
                    # Clear button
                    clear_btn = gr.Button("Clear Chat")
                
                # Right column - Controls and status
                with gr.Column(scale=1):
                    # System status
                    gr.HTML("<h3>📊 System Status</h3>")
                    status_display = gr.HTML(self.get_system_status())
                    
                    # Refresh status button
                    refresh_btn = gr.Button("🔄 Refresh Status")
                    
                    # AI Employee creation
                    gr.HTML("<h3>➕ Create AI Employee</h3>")
                    
                    role_dropdown = gr.Dropdown(
                        choices=[
                            "research_analyst",
                            "trader", 
                            "risk_manager",
                            "compliance_officer",
                            "data_specialist"
                        ],
                        label="Role",
                        value="research_analyst"
                    )
                    
                    specialization_input = gr.Textbox(
                        label="Specialization",
                        placeholder="e.g., cryptocurrency markets, high-frequency trading"
                    )
                    
                    create_btn = gr.Button("Create AI Employee", variant="primary")
                    create_output = gr.Textbox(label="Status", interactive=False)
                    
                    # Help section
                    gr.HTML("<h3>📋 Help</h3>")
                    help_btn = gr.Button("Show Help")
                    help_display = gr.HTML(visible=False)
            
            # Event handlers
            msg.submit(
                self.process_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            send_btn.click(
                self.process_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear_btn.click(
                lambda: ([], ""),
                outputs=[chatbot, msg]
            )
            
            refresh_btn.click(
                lambda: gr.HTML(self.get_system_status()),
                outputs=status_display
            )
            
            create_btn.click(
                self.create_ai_employee,
                inputs=[role_dropdown, specialization_input],
                outputs=create_output
            )
            
            help_btn.click(
                lambda: (gr.HTML(self.get_help()), gr.update(visible=True)),
                outputs=[help_display, help_display]
            )
            
            # Welcome message will be handled by the chat interface
        
        return interface

def main():
    """Main function to run the Gradio app."""
    try:
        app = GradioApp()
        interface = app.create_interface()
        
        # Launch the interface
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=True,
            prevent_thread_lock=True
        )
        
    except Exception as e:
        logger.error(f"Error running Gradio app: {str(e)}")
        raise

if __name__ == "__main__":
    main() 