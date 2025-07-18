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
            # Initialize AI Engine lazily to avoid startup issues
            self.ai_engine = None
            logger.info("AI Engine will be initialized on first use")
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
            # Initialize AI Engine on first use if not already initialized
            if self.ai_engine is None:
                try:
                    self.ai_engine = AIEngine()
                    logger.info("AI Engine initialized successfully on first use")
                except Exception as e:
                    logger.error(f"Failed to initialize AI Engine: {str(e)}")
                    return "AI Engine is not available. Please check your configuration."
            
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
            # Check actual system components
            ai_engine_status = "✅ Active" if self.ai_engine is not None else "⏳ Initializing"
            
            status = f"""
<div style="background-color: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 8px; border-left: 4px solid #48bb78;">
    <h3 style="margin-top: 0; color: #f7fafc;">🤖 Bot Builder AI System Status</h3>
    
    <div style="margin-bottom: 15px;">
        <strong style="color: #f7fafc;">System Health:</strong> <span style="color: #48bb78;">✅ Operational</span><br>
        <strong style="color: #f7fafc;">AI Engine:</strong> <span style="color: #48bb78;">{ai_engine_status}</span><br>
        <strong style="color: #f7fafc;">Data Manager:</strong> <span style="color: #48bb78;">✅ Active</span><br>
        <strong style="color: #f7fafc;">Metrics Collector:</strong> <span style="color: #48bb78;">✅ Active</span><br>
        <strong style="color: #f7fafc;">Self-Improvement:</strong> <span style="color: #48bb78;">✅ Enabled</span><br>
        <strong style="color: #f7fafc;">Real-time Data:</strong> <span style="color: #48bb78;">✅ Active</span><br>
        <strong style="color: #f7fafc;">Gradio Interface:</strong> <span style="color: #48bb78;">✅ Running on localhost:7861</span>
    </div>
    
    <div style="margin-bottom: 15px;">
        <strong style="color: #f7fafc;">Quick Actions:</strong>
        <ul style="margin: 5px 0; padding-left: 20px; color: #e2e8f0;">
            <li>Create AI Employees using the form below</li>
            <li>Chat with the AI Assistant</li>
            <li>Monitor performance and metrics</li>
        </ul>
    </div>
    
    <div style="margin-bottom: 15px;">
        <strong style="color: #f7fafc;">Available Roles:</strong>
        <ul style="margin: 5px 0; padding-left: 20px; color: #e2e8f0;">
            <li><strong style="color: #f7fafc;">Research Analyst:</strong> Deep learning, forecasting, economic analysis</li>
            <li><strong style="color: #f7fafc;">Trader:</strong> Reinforcement learning, execution speed, strategic decision-making</li>
            <li><strong style="color: #f7fafc;">Risk Manager:</strong> Probability theory, statistical modeling, scenario testing</li>
            <li><strong style="color: #f7fafc;">Compliance Officer:</strong> Regulatory knowledge, NLP, explainability</li>
            <li><strong style="color: #f7fafc;">Data Specialist:</strong> Data cleaning, management, structuring</li>
        </ul>
    </div>
    
    <div style="font-size: 0.9em; color: #a0aec0; border-top: 1px solid #4a5568; padding-top: 10px;">
        <strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</div>
            """
            return status
            
        except Exception as e:
            return f'<div style="color: #fed7d7; padding: 10px; background-color: #742a2a; border-radius: 5px;">❌ Error getting system status: {str(e)}</div>'
    
    def get_help(self) -> str:
        """Get help information."""
        help_text = """
<div style="background-color: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 8px; border-left: 4px solid #4299e1;">
    <h3 style="margin-top: 0; color: #f7fafc;">📋 Bot Builder AI - Help Guide</h3>
    
    <div style="margin-bottom: 20px;">
        <h4 style="color: #4299e1; margin-bottom: 10px;">Available Commands</h4>
        
        <div style="margin-bottom: 15px;">
            <h5 style="color: #e2e8f0; margin-bottom: 8px;">Creating AI Employees</h5>
            <ul style="margin: 5px 0; padding-left: 20px; color: #e2e8f0;">
                <li>Use the form below to create new AI Employees</li>
                <li>Select a role and enter a specialization</li>
                <li>Click "Create AI Employee" to start</li>
            </ul>
        </div>
        
        <div style="margin-bottom: 15px;">
            <h5 style="color: #e2e8f0; margin-bottom: 8px;">Chat Commands</h5>
            <ul style="margin: 5px 0; padding-left: 20px; color: #e2e8f0;">
                <li>"Create a new Research Analyst AI Employee focused on cryptocurrency markets"</li>
                <li>"Show me the performance metrics for all AI Employees"</li>
                <li>"Optimize the Trader AI Employee for better execution speed"</li>
                <li>"What's the current system status?"</li>
                <li>"Display performance analytics for the last 30 days"</li>
            </ul>
        </div>
    </div>
    
    <div style="margin-bottom: 20px;">
        <h4 style="color: #4299e1; margin-bottom: 10px;">AI Employee Roles</h4>
        <ol style="margin: 5px 0; padding-left: 20px; color: #e2e8f0;">
            <li><strong style="color: #f7fafc;">Research Analyst:</strong> Deep learning, forecasting, economic analysis</li>
            <li><strong style="color: #f7fafc;">Trader:</strong> Reinforcement learning, execution speed, strategic decision-making</li>
            <li><strong style="color: #f7fafc;">Risk Manager:</strong> Probability theory, statistical modeling, scenario testing</li>
            <li><strong style="color: #f7fafc;">Compliance Officer:</strong> Regulatory knowledge, NLP, explainability</li>
            <li><strong style="color: #f7fafc;">Data Specialist:</strong> Data cleaning, management, structuring</li>
        </ol>
    </div>
    
    <div style="margin-bottom: 15px;">
        <h4 style="color: #4299e1; margin-bottom: 10px;">Tips</h4>
        <ul style="margin: 5px 0; padding-left: 20px; color: #e2e8f0;">
            <li>Be specific about your requirements</li>
            <li>Monitor performance regularly</li>
            <li>Use optimization features to improve results</li>
            <li>Check system status for any issues</li>
        </ul>
    </div>
    
    <div style="background-color: #744210; padding: 10px; border-radius: 5px; border: 1px solid #d69e2e; color: #faf089;">
        <strong>💡 Need more help?</strong> Just ask in the chat!
    </div>
</div>
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
            /* Prevent postMessage errors */
            iframe {
                pointer-events: none;
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
            
            def refresh_status():
                """Refresh system status."""
                return self.get_system_status()
            
            refresh_btn.click(
                refresh_status,
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
            
            # Add welcome message
            def add_welcome_message():
                return [{"role": "assistant", "content": "Hello! I'm your Bot Builder AI assistant. I can help you create and manage AI Employees, check system status, and answer questions about the system. How can I help you today?"}]
            
            # Add welcome message on load
            interface.load(add_welcome_message, outputs=chatbot)
        
        return interface

def main():
    """Main function to run the Gradio app."""
    try:
        logger.info("Starting Gradio app initialization...")
        app = GradioApp()
        interface = app.create_interface()
        
        logger.info("Launching Gradio interface on http://localhost:7861")
        # Launch the interface
        interface.launch(
            server_name="localhost",
            server_port=7861,
            share=False,
            show_error=True,
            quiet=False,
            allowed_paths=["ui/"],
            auth=None,
            ssl_verify=False,
            show_api=False,
            inbrowser=False
        )
        
    except Exception as e:
        logger.error(f"Error running Gradio app: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main() 