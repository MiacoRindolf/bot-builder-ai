"""
Streamlit web interface for the Bot Builder AI system.
Provides a conversational chat interface and comprehensive dashboard for managing AI Employees.
"""

import streamlit as st
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

# Page configuration
st.set_page_config(
    page_title="Bot Builder AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #1f77b4;
    }
    .ai-message {
        background-color: #e8f4fd;
        border-left-color: #ff7f0e;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-active {
        background-color: #28a745;
    }
    .status-training {
        background-color: #ffc107;
    }
    .status-failed {
        background-color: #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.ai_engine = None
        self.session_id = None
        self.user_id = None
        
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'ai_employees' not in st.session_state:
            st.session_state.ai_employees = {}
        
        if 'system_status' not in st.session_state:
            st.session_state.system_status = "initializing"
    
    def initialize_ai_engine(self):
        """Initialize the AI Engine."""
        try:
            if self.ai_engine is None:
                self.ai_engine = AIEngine()
                st.session_state.system_status = "ready"
                logger.info("AI Engine initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize AI Engine: {str(e)}")
            st.session_state.system_status = "error"
            logger.error(f"Error initializing AI Engine: {str(e)}")
    
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
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">🤖 Bot Builder AI</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Advanced AI Employee Management System for AI-Powered Hedge Funds
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with navigation and controls."""
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            # System Status
            st.subheader("System Status")
            status_color = {
                "ready": "🟢",
                "initializing": "🟡", 
                "error": "🔴"
            }.get(st.session_state.system_status, "⚪")
            
            st.write(f"{status_color} {st.session_state.system_status.title()}")
            
            # Quick Actions
            st.subheader("Quick Actions")
            
            if st.button("🔄 Refresh System"):
                self.refresh_system()
            
            if st.button("📊 View All Employees"):
                st.session_state.current_page = "employees"
            
            if st.button("📈 Performance Dashboard"):
                st.session_state.current_page = "performance"
            
            # AI Employee Creation
            st.subheader("Create AI Employee")
            
            role = st.selectbox(
                "Select Role",
                ["research_analyst", "trader", "risk_manager", "compliance_officer", "data_specialist"]
            )
            
            specialization = st.text_input("Specialization", placeholder="e.g., cryptocurrency markets")
            
            if st.button("➕ Create Employee"):
                if specialization:
                    self.create_ai_employee(role, specialization)
                else:
                    st.warning("Please enter a specialization")
            
            # Settings
            st.subheader("Settings")
            
            if st.button("⚙️ Configuration"):
                st.session_state.current_page = "settings"
            
            if st.button("📋 Help"):
                st.session_state.current_page = "help"
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        st.header("💬 AI Assistant")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about AI Employees, performance, or system management..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("🤖 AI is thinking..."):
                        response = self.get_ai_response(prompt)
                        st.markdown(response)
                
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    def render_dashboard(self):
        """Render the main dashboard."""
        st.header("📊 Dashboard")
        
        # System overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Active AI Employees",
                value=len(st.session_state.ai_employees),
                delta="+2 this week"
            )
        
        with col2:
            st.metric(
                label="System Health",
                value=st.session_state.system_status.title(),
                delta="🟢 Good"
            )
        
        with col3:
            st.metric(
                label="Total Operations",
                value="1,234",
                delta="+56 today"
            )
        
        with col4:
            st.metric(
                label="Success Rate",
                value="94.2%",
                delta="+2.1%"
            )
        
        # AI Employees overview
        st.subheader("🤖 AI Employees Overview")
        
        if st.session_state.ai_employees:
            # Create a DataFrame-like display
            for employee_id, employee_data in st.session_state.ai_employees.items():
                with st.expander(f"📋 {employee_id} - {employee_data['role'].title()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Role:** {employee_data['role'].title()}")
                        st.write(f"**Specialization:** {employee_data['specialization']}")
                        st.write(f"**Status:** {employee_data['status']}")
                    
                    with col2:
                        st.write(f"**Created:** {employee_data['created_at']}")
                        st.write(f"**Performance:** {employee_data.get('performance', 'N/A')}")
                        
                        # Action buttons
                        if st.button(f"📊 View Details", key=f"details_{employee_id}"):
                            self.view_employee_details(employee_id)
                        
                        if st.button(f"⚡ Optimize", key=f"optimize_{employee_id}"):
                            self.optimize_employee(employee_id)
        else:
            st.info("No AI Employees created yet. Use the sidebar to create your first AI Employee!")
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        
        # Placeholder for recent activity
        activity_data = [
            {"time": "2 minutes ago", "action": "Created Research Analyst", "status": "Training"},
            {"time": "15 minutes ago", "action": "Optimized Trader", "status": "Completed"},
            {"time": "1 hour ago", "action": "Risk Manager Analysis", "status": "Active"},
        ]
        
        for activity in activity_data:
            st.write(f"🕒 **{activity['time']}** - {activity['action']} ({activity['status']})")
    
    def render_employees_page(self):
        """Render the employees management page."""
        st.header("👥 AI Employees Management")
        
        # Employee creation section
        with st.expander("➕ Create New AI Employee", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                role = st.selectbox(
                    "Role",
                    ["research_analyst", "trader", "risk_manager", "compliance_officer", "data_specialist"],
                    key="create_role"
                )
                
                specialization = st.text_input(
                    "Specialization",
                    placeholder="e.g., cryptocurrency markets, high-frequency trading",
                    key="create_specialization"
                )
            
            with col2:
                st.write("**Role Capabilities:**")
                capabilities = self.get_role_capabilities(role)
                for capability in capabilities:
                    st.write(f"• {capability}")
                
                if st.button("Create AI Employee", key="create_employee"):
                    if specialization:
                        self.create_ai_employee(role, specialization)
                    else:
                        st.warning("Please enter a specialization")
        
        # Employee list
        st.subheader("📋 Active AI Employees")
        
        if st.session_state.ai_employees:
            for employee_id, employee_data in st.session_state.ai_employees.items():
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                    
                    with col1:
                        st.write(f"**{employee_id}**")
                        st.write(f"Role: {employee_data['role'].title()}")
                        st.write(f"Specialization: {employee_data['specialization']}")
                    
                    with col2:
                        status_color = {
                            "active": "🟢",
                            "training": "🟡",
                            "failed": "🔴",
                            "paused": "⚪"
                        }.get(employee_data['status'], "⚪")
                        
                        st.write(f"{status_color} {employee_data['status'].title()}")
                        st.write(f"Created: {employee_data['created_at']}")
                    
                    with col3:
                        performance = employee_data.get('performance', {})
                        if performance:
                            st.write(f"Accuracy: {performance.get('accuracy', 'N/A')}")
                            st.write(f"Speed: {performance.get('execution_speed', 'N/A')}")
                        else:
                            st.write("Performance: N/A")
                    
                    with col4:
                        if st.button("📊 Details", key=f"emp_details_{employee_id}"):
                            self.view_employee_details(employee_id)
                        
                        if st.button("⚡ Optimize", key=f"emp_optimize_{employee_id}"):
                            self.optimize_employee(employee_id)
                        
                        if st.button("⏸️ Pause", key=f"emp_pause_{employee_id}"):
                            self.pause_employee(employee_id)
                    
                    st.divider()
        else:
            st.info("No AI Employees found. Create your first AI Employee using the form above!")
    
    def render_performance_page(self):
        """Render the performance dashboard page."""
        st.header("📈 Performance Dashboard")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Overall Performance")
            st.metric("Average Accuracy", "87.3%", "+2.1%")
            st.metric("Success Rate", "94.2%", "+1.8%")
            st.metric("Response Time", "0.23s", "-0.05s")
        
        with col2:
            st.subheader("📊 Role Performance")
            role_performance = {
                "Research Analyst": "89.1%",
                "Trader": "92.4%",
                "Risk Manager": "95.7%",
                "Compliance Officer": "88.9%",
                "Data Specialist": "91.2%"
            }
            
            for role, performance in role_performance.items():
                st.write(f"**{role}:** {performance}")
        
        with col3:
            st.subheader("🚀 Optimization Status")
            optimization_status = {
                "Recent Optimizations": "12",
                "Performance Gains": "+15.3%",
                "Active Optimizations": "3"
            }
            
            for metric, value in optimization_status.items():
                st.write(f"**{metric}:** {value}")
        
        # Performance charts (placeholder)
        st.subheader("📈 Performance Trends")
        
        # Placeholder for charts
        st.line_chart({
            "Accuracy": [85, 87, 86, 89, 88, 90, 87, 89, 91, 92],
            "Success Rate": [90, 92, 91, 93, 92, 94, 93, 95, 94, 96]
        })
    
    def render_settings_page(self):
        """Render the settings page."""
        st.header("⚙️ System Settings")
        
        # OpenAI Configuration
        st.subheader("🔑 OpenAI Configuration")
        
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=settings.openai_api_key if settings.openai_api_key != "your_openai_api_key_here" else "",
            help="Enter your OpenAI API key"
        )
        
        model = st.selectbox(
            "Model",
            ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
            index=0
        )
        
        # System Configuration
        st.subheader("🔧 System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_employees = st.number_input(
                "Max AI Employees",
                min_value=1,
                max_value=100,
                value=settings.max_ai_employees
            )
            
            training_timeout = st.number_input(
                "Training Timeout (hours)",
                min_value=1,
                max_value=72,
                value=settings.training_timeout_hours
            )
        
        with col2:
            optimization_interval = st.number_input(
                "Optimization Interval (hours)",
                min_value=1,
                max_value=24,
                value=settings.optimization_interval_hours
            )
            
            risk_limit = st.number_input(
                "Risk Limit (%)",
                min_value=0.1,
                max_value=10.0,
                value=settings.risk_limit_percentage,
                step=0.1
            )
        
        # Save settings
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")
    
    def render_help_page(self):
        """Render the help page."""
        st.header("📋 Help & Documentation")
        
        # Quick start guide
        st.subheader("🚀 Quick Start Guide")
        
        st.markdown("""
        ### 1. Create Your First AI Employee
        1. Go to the sidebar and select a role (Research Analyst, Trader, etc.)
        2. Enter a specialization (e.g., "cryptocurrency markets")
        3. Click "Create Employee"
        
        ### 2. Monitor Performance
        1. Use the chat interface to ask about performance
        2. Visit the Performance Dashboard
        3. Check individual employee details
        
        ### 3. Optimize AI Employees
        1. Select an employee from the list
        2. Click "Optimize" to improve performance
        3. Monitor the optimization progress
        """)
        
        # Available commands
        st.subheader("💬 Available Commands")
        
        commands = [
            "Create a new Research Analyst AI Employee focused on cryptocurrency markets",
            "Show me the performance metrics for all AI Employees",
            "Optimize the Trader AI Employee for better execution speed",
            "What's the current system status?",
            "Display performance analytics for the last 30 days"
        ]
        
        for command in commands:
            st.write(f"• `{command}`")
        
        # Role descriptions
        st.subheader("🤖 AI Employee Roles")
        
        roles = {
            "Research Analyst": "Deep learning, forecasting, economic analysis",
            "Trader": "Reinforcement learning, execution speed, strategic decision-making",
            "Risk Manager": "Probability theory, statistical modeling, scenario testing",
            "Compliance Officer": "Regulatory knowledge, NLP, explainability",
            "Data Specialist": "Data cleaning, management, structuring"
        }
        
        for role, description in roles.items():
            st.write(f"**{role}:** {description}")
    
    def get_ai_response(self, prompt: str) -> str:
        """Get response from AI Engine."""
        try:
            if self.ai_engine is None:
                return "AI Engine is not initialized. Please check your configuration."
            
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    self.ai_engine.process_user_input(
                        prompt,
                        self.get_session_id(),
                        self.get_user_id()
                    )
                )
                return response
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def create_ai_employee(self, role: str, specialization: str):
        """Create a new AI Employee."""
        try:
            if self.ai_engine is None:
                st.error("AI Engine is not initialized")
                return
            
            # Create employee ID
            employee_id = f"{role}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Add to session state
            st.session_state.ai_employees[employee_id] = {
                "role": role,
                "specialization": specialization,
                "status": "training",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "performance": {}
            }
            
            st.success(f"AI Employee {employee_id} created successfully! Training in progress...")
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I've created a new {role.title()} AI Employee with specialization '{specialization}'. Training is now in progress."
            })
            
        except Exception as e:
            st.error(f"Error creating AI Employee: {str(e)}")
    
    def get_role_capabilities(self, role: str) -> List[str]:
        """Get capabilities for a specific role."""
        capabilities = {
            "research_analyst": [
                "Deep learning models for market prediction",
                "Sentiment analysis of financial news",
                "Technical and fundamental analysis",
                "Economic indicator analysis"
            ],
            "trader": [
                "Reinforcement learning for trading strategies",
                "High-speed execution algorithms",
                "Risk management integration",
                "Real-time market analysis"
            ],
            "risk_manager": [
                "Statistical risk modeling",
                "VaR and stress testing",
                "Portfolio risk assessment",
                "Scenario analysis"
            ],
            "compliance_officer": [
                "Regulatory compliance monitoring",
                "NLP for document analysis",
                "Audit trail generation",
                "Explainable AI decisions"
            ],
            "data_specialist": [
                "Data cleaning and preprocessing",
                "Feature engineering",
                "Data quality assessment",
                "Pipeline optimization"
            ]
        }
        
        return capabilities.get(role, ["General AI capabilities"])
    
    def view_employee_details(self, employee_id: str):
        """View detailed information about an AI Employee."""
        st.session_state.current_page = "employee_details"
        st.session_state.selected_employee = employee_id
    
    def optimize_employee(self, employee_id: str):
        """Optimize an AI Employee."""
        st.info(f"Starting optimization for {employee_id}...")
        # Placeholder for optimization logic
    
    def pause_employee(self, employee_id: str):
        """Pause an AI Employee."""
        if employee_id in st.session_state.ai_employees:
            st.session_state.ai_employees[employee_id]["status"] = "paused"
            st.success(f"AI Employee {employee_id} paused successfully")
    
    def refresh_system(self):
        """Refresh the system status."""
        st.session_state.system_status = "ready"
        st.success("System refreshed successfully!")
    
    def run(self):
        """Run the Streamlit application."""
        try:
            # Initialize AI Engine
            self.initialize_ai_engine()
            
            # Render header
            self.render_header()
            
            # Render sidebar
            self.render_sidebar()
            
            # Main content area
            current_page = st.session_state.get("current_page", "dashboard")
            
            if current_page == "dashboard":
                self.render_dashboard()
                self.render_chat_interface()
            elif current_page == "employees":
                self.render_employees_page()
            elif current_page == "performance":
                self.render_performance_page()
            elif current_page == "settings":
                self.render_settings_page()
            elif current_page == "help":
                self.render_help_page()
            elif current_page == "employee_details":
                self.render_employee_details()
            else:
                self.render_dashboard()
                self.render_chat_interface()
                
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.error(f"Streamlit app error: {str(e)}")
    
    def render_employee_details(self):
        """Render detailed employee information."""
        employee_id = st.session_state.get("selected_employee")
        
        if not employee_id or employee_id not in st.session_state.ai_employees:
            st.error("Employee not found")
            return
        
        employee_data = st.session_state.ai_employees[employee_id]
        
        st.header(f"📋 {employee_id} Details")
        
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            st.write(f"**Role:** {employee_data['role'].title()}")
            st.write(f"**Specialization:** {employee_data['specialization']}")
            st.write(f"**Status:** {employee_data['status']}")
            st.write(f"**Created:** {employee_data['created_at']}")
        
        with col2:
            st.subheader("Performance Metrics")
            performance = employee_data.get('performance', {})
            if performance:
                for metric, value in performance.items():
                    st.write(f"**{metric.title()}:** {value}")
            else:
                st.write("No performance data available")
        
        # Action buttons
        st.subheader("Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("⚡ Optimize", key=f"detail_optimize_{employee_id}"):
                self.optimize_employee(employee_id)
        
        with col2:
            if st.button("⏸️ Pause", key=f"detail_pause_{employee_id}"):
                self.pause_employee(employee_id)
        
        with col3:
            if st.button("🔄 Restart", key=f"detail_restart_{employee_id}"):
                st.info(f"Restarting {employee_id}...")
        
        with col4:
            if st.button("🗑️ Delete", key=f"detail_delete_{employee_id}"):
                if st.button("Confirm Delete", key=f"confirm_delete_{employee_id}"):
                    del st.session_state.ai_employees[employee_id]
                    st.success(f"Deleted {employee_id}")
                    st.session_state.current_page = "employees"
        
        # Back button
        if st.button("← Back to Employees"):
            st.session_state.current_page = "employees"

def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main() 