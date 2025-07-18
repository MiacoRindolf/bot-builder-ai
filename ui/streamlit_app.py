import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
from ui.styles import MAIN_STYLES
from core.roles import ROLES
from core.ai_engine import AIEngine
import uuid

# Page config
st.set_page_config(
    page_title="Bot Builder AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS
st.markdown(MAIN_STYLES, unsafe_allow_html=True)

# Initialize AI Engine (singleton)
if "ai_engine" not in st.session_state:
    st.session_state["ai_engine"] = AIEngine()

# Initialize chat messages
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

# --- Left Sidebar (Navigation) ---
def render_left_sidebar():
    st.markdown("### Control Panel")
    st.markdown("---")
    if st.button("Dashboard", key="dashboard_btn"):
        st.session_state["page"] = "dashboard"
    if st.button("Employees", key="employees_btn"):
        st.session_state["page"] = "employees"
    if st.button("Performance", key="performance_btn"):
        st.session_state["page"] = "performance"
    if st.button("Settings", key="settings_btn"):
        st.session_state["page"] = "settings"
    if st.button("Help", key="help_btn"):
        st.session_state["page"] = "help"

# --- Right Sidebar (AI Assistant) ---
def render_right_sidebar():
    # Create a container for the right sidebar
    with st.container():
        st.markdown('<div class="right-sidebar-chat">', unsafe_allow_html=True)
        st.markdown("### AI Assistant")
        
        # Chat messages area with scrolling
        st.markdown('<div class="chat-messages-container">', unsafe_allow_html=True)
        if st.session_state["chat_messages"]:
            for message in st.session_state["chat_messages"]:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="ai-message">{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Message", placeholder="Ask me anything...", key="chat_input", label_visibility="collapsed")
            submitted = st.form_submit_button("Send", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle message sending
        if submitted and user_input:
            st.session_state["chat_messages"].append({"role": "user", "content": user_input})
            ai_response = "[Error: AI engine not available]"
            try:
                session_id = st.session_state.get("chat_session_id")
                if not session_id:
                    session_id = str(uuid.uuid4())
                    st.session_state["chat_session_id"] = session_id
                user_id = st.session_state.get("chat_user_id")
                if not user_id:
                    user_id = f"user_{uuid.uuid4().hex[:8]}"
                    st.session_state["chat_user_id"] = user_id
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                ai_response = loop.run_until_complete(
                    st.session_state["ai_engine"].process_user_input(user_input, session_id, user_id)
                )
                loop.close()
            except Exception as e:
                ai_response = f"[AI error: {e}]"
            st.session_state["chat_messages"].append({"role": "assistant", "content": ai_response})
            st.rerun()

# --- Header ---
def render_header():
    st.markdown("""
        <div style="text-align:center; margin-bottom:2rem;">
            <h1>Bot Builder AI</h1>
            <p>Advanced AI Employee Management System</p>
        </div>
    """, unsafe_allow_html=True)

# --- Dashboard Page ---
def render_dashboard():
    st.markdown("# Dashboard")
    
    # System Status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>System Status</h3>
            <p>Healthy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        employee_count = len(st.session_state.get("employees", []))
        st.markdown(f"""
        <div class="metric-card">
            <h3>AI Employees</h3>
            <p>{employee_count}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>API Calls</h3>
            <p>1,234</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Success Rate</h3>
            <p>94.2%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent Employees
    st.markdown("## Recent AI Employees")
    employees = st.session_state.get("employees", [])
    if employees:
        for emp in employees[-3:]:  # Show last 3
            st.markdown(f"""
            <div class="metric-card">
                <h4>{emp['role_name']}</h4>
                <p>{emp['specialization']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No AI Employees created yet.")

# --- Employees Page ---
def render_employees():
    st.markdown("# Employees")
    if "employees" not in st.session_state:
        st.session_state["employees"] = []

    # List employees
    if st.session_state["employees"]:
        for emp in st.session_state["employees"]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{emp['role_name']}</h4>
                <p>{emp['specialization']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No AI Employees created yet.")

    st.markdown("---")
    st.subheader("Create New AI Employee")
    with st.form("create_employee_form"):
        role_options = [(r["name"], r["key"]) for r in ROLES]
        role_name = st.selectbox("Role", [r[0] for r in role_options])
        specialization = st.text_input("Specialization", "")
        submitted = st.form_submit_button("Create Employee")
        if submitted:
            role_key = dict(role_options)[role_name]
            st.session_state["employees"].append({
                "role_key": role_key,
                "role_name": role_name,
                "specialization": specialization
            })
            st.success(f"Created {role_name} ({specialization})")

# --- Performance Page ---
def render_performance():
    st.markdown("# Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>System Performance</h3>
            <p>CPU Usage: 23%</p>
            <p>Memory Usage: 45%</p>
            <p>API Calls: 1,234</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>AI Performance</h3>
            <p>Response Time: 1.2s</p>
            <p>Accuracy: 94.2%</p>
            <p>Success Rate: 98.7%</p>
        </div>
        """, unsafe_allow_html=True)

# --- Settings Page ---
def render_settings():
    st.markdown("# Settings")
    
    st.markdown("""
        <div class="metric-card">
            <h3>General Settings</h3>
            <p>Theme: Dark</p>
            <p>Language: English</p>
            <p>Timezone: UTC</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("API Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", value="sk-...")
    model = st.selectbox("Model", ["gpt-35turbo", "gpt-4", "gpt-4urbo"]) 
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

# --- Help Page ---
def render_help():
    st.markdown("# Help & Documentation")
    
    st.markdown("""
        <div class="metric-card">
            <h3>Getting Started</h3>
            <p>1. Create your first AI Employee using the sidebar</p>
            <p>2. Chat with the AI Assistant for guidance</p>
            <p>3. Monitor performance in the dashboard</p>
            <p>4. Optimize your AI Employees for better results</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("AI Employee Roles")
    
    for role in ROLES:
        st.markdown(f"""
        <div class="metric-card">
            <h4>{role['name']}</h4>
            <p>{role['description']}</p>
        </div>
        """, unsafe_allow_html=True)

# --- Main Content Routing ---
def render_main():
    page = st.session_state.get("page", "dashboard")
    if page == "dashboard":
        render_dashboard()
    elif page == "employees":
        render_employees()
    elif page == "performance":
        render_performance()
    elif page == "settings":
        render_settings()
    elif page == "help":
        render_help()
    else:
        st.markdown("## Page not found")

# --- Main App ---
def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "dashboard"
    
    # Create three-column layout
    left_col, main_col, right_col = st.columns([1, 3, 1])
    
    # Left sidebar (navigation)
    with left_col:
        render_left_sidebar()
    
    # Main content area
    with main_col:
        render_header()
        render_main()
    
    # Right sidebar (AI Assistant)
    with right_col:
        render_right_sidebar()

if __name__ == "__main__":
    main() 