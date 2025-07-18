import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
import asyncio
import uuid
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.styles import MAIN_STYLES
from core.roles import ROLES
from core.ai_engine import AIEngine
from core.self_improvement_engine import SelfImprovementEngine
from monitoring.metrics import MetricsCollector

# Page config
st.set_page_config(
    page_title="Bot Builder AI - CEO Portal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS
st.markdown(MAIN_STYLES, unsafe_allow_html=True)

# Initialize AI Engine (singleton)
if "ai_engine" not in st.session_state:
    st.session_state["ai_engine"] = AIEngine()

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "dashboard"
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "employees" not in st.session_state:
    st.session_state["employees"] = []
if "system_health" not in st.session_state:
    st.session_state["system_health"] = "healthy"
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = datetime.now()

# Custom CSS for CEO Portal
st.markdown("""
<style>
.ceo-header {
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #1f77b4;
    margin-bottom: 1rem;
}

.metric-card h3 {
    color: #1f77b4;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #2c3e50;
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-healthy { background-color: #27ae60; }
.status-warning { background-color: #f39c12; }
.status-error { background-color: #e74c3c; }

.employee-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
    border-left: 4px solid #3498db;
}

.employee-card h4 {
    color: #2c3e50;
    margin-bottom: 0.5rem;
}

.chat-container {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    height: 400px;
    overflow-y: auto;
}

.user-message {
    background: #007bff;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 15px;
    margin: 0.5rem 0;
    text-align: right;
}

.ai-message {
    background: #e9ecef;
    color: #2c3e50;
    padding: 0.5rem 1rem;
    border-radius: 15px;
    margin: 0.5rem 0;
    text-align: left;
}

.sidebar-nav {
    background: #2c3e50;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.sidebar-nav h3 {
    color: white;
    margin-bottom: 1rem;
}

.nav-button {
    width: 100%;
    margin-bottom: 0.5rem;
    background: #34495e;
    color: white;
    border: none;
    padding: 0.75rem;
    border-radius: 5px;
    text-align: left;
}

.nav-button:hover {
    background: #1f77b4;
}

.proposal-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
    border-left: 4px solid #e74c3c;
}

.proposal-card.pending {
    border-left-color: #f39c12;
}

.proposal-card.approved {
    border-left-color: #27ae60;
}

.proposal-card.rejected {
    border-left-color: #e74c3c;
}
</style>
""", unsafe_allow_html=True)

def render_header():
    """Render the CEO Portal header."""
    st.markdown("""
    <div class="ceo-header">
        <h1>🤖 Bot Builder AI - CEO Portal</h1>
        <p>Advanced AI Employee Management System for Autonomous Hedge Fund Operations</p>
        <p><strong>True AI Self-Improvement • Real-time Market Data • Advanced RL • Explainability</strong></p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the navigation sidebar."""
    st.sidebar.markdown("""
    <div class="sidebar-nav">
        <h3>🎯 CEO Control Panel</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    if st.sidebar.button("📊 Executive Dashboard", key="dashboard_btn", use_container_width=True):
        st.session_state["current_page"] = "dashboard"
    
    if st.sidebar.button("👥 AI Employee Management", key="employees_btn", use_container_width=True):
        st.session_state["current_page"] = "employees"
    
    if st.sidebar.button("📈 Performance Analytics", key="performance_btn", use_container_width=True):
        st.session_state["current_page"] = "performance"
    
    if st.sidebar.button("🧠 Self-Improvement Hub", key="self_improvement_btn", use_container_width=True):
        st.session_state["current_page"] = "self_improvement"
    
    if st.sidebar.button("📊 Real-time Market Data", key="market_data_btn", use_container_width=True):
        st.session_state["current_page"] = "market_data"
    
    if st.sidebar.button("🔧 System Configuration", key="settings_btn", use_container_width=True):
        st.session_state["current_page"] = "settings"
    
    if st.sidebar.button("📋 Help & Documentation", key="help_btn", use_container_width=True):
        st.session_state["current_page"] = "help"
    
    st.sidebar.markdown("---")
    
    # Quick Actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Refresh System Status", key="refresh_btn", use_container_width=True):
        st.session_state["last_refresh"] = datetime.now()
        st.rerun()
    
    if st.sidebar.button("📊 Generate Performance Report", key="report_btn", use_container_width=True):
        st.session_state["current_page"] = "performance"
    
    if st.sidebar.button("🧠 Analyze System", key="analyze_btn", use_container_width=True):
        st.session_state["current_page"] = "self_improvement"
    
    # System Status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 System Status")
    
    status_color = {
        "healthy": "status-healthy",
        "warning": "status-warning", 
        "error": "status-error"
    }
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <span class="status-indicator {status_color[st.session_state['system_health']]}"></span>
        <strong>System Health:</strong> {st.session_state['system_health'].title()}
    </div>
    """, unsafe_allow_html=True)
    
    employee_count = len(st.session_state.get("employees", []))
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>Active AI Employees:</strong> {employee_count}
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>Last Updated:</strong> {st.session_state['last_refresh'].strftime('%H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

def get_system_metrics():
    """Get system metrics for dashboard."""
    try:
        # This would normally come from the metrics collector
        return {
            "total_employees": len(st.session_state.get("employees", [])),
            "active_employees": len([e for e in st.session_state.get("employees", []) if e.get("status") == "active"]),
            "system_uptime": "99.8%",
            "api_calls_today": 1247,
            "success_rate": 94.2,
            "avg_response_time": "1.2s",
            "monthly_spend": 12.45,
            "spend_limit": 15.0,
            "self_improvement_proposals": 3,
            "pending_approvals": 1
        }
    except Exception as e:
        st.error(f"Error getting system metrics: {str(e)}")
        return {}

def render_dashboard():
    """Render the executive dashboard."""
    st.markdown("## 📊 Executive Dashboard")
    
    # Get system metrics
    metrics = get_system_metrics()
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🤖 AI Employees</h3>
            <div class="metric-value">{metrics.get('total_employees', 0)}</div>
            <p>Active: {metrics.get('active_employees', 0)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Success Rate</h3>
            <div class="metric-value">{metrics.get('success_rate', 0)}%</div>
            <p>Avg Response: {metrics.get('avg_response_time', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Monthly Spend</h3>
            <div class="metric-value">${metrics.get('monthly_spend', 0):.2f}</div>
            <p>Limit: ${metrics.get('spend_limit', 0):.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🧠 Self-Improvement</h3>
            <div class="metric-value">{metrics.get('self_improvement_proposals', 0)}</div>
            <p>Pending: {metrics.get('pending_approvals', 0)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System Health and Performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 System Health")
        
        # System health chart
        health_data = {
            'Component': ['AI Engine', 'Data Manager', 'RL Engine', 'Market Data', 'Explainability'],
            'Status': ['Healthy', 'Healthy', 'Healthy', 'Warning', 'Healthy'],
            'Performance': [95, 88, 92, 75, 96]
        }
        df_health = pd.DataFrame(health_data)
        
        fig = px.bar(df_health, x='Component', y='Performance', 
                    color='Status', color_discrete_map={
                        'Healthy': '#27ae60', 'Warning': '#f39c12', 'Error': '#e74c3c'
                    })
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 API Usage (Last 24h)")
        
        # API usage chart
        hours = list(range(24))
        api_calls = [45, 52, 38, 29, 23, 18, 25, 67, 89, 124, 156, 178, 
                    145, 167, 189, 201, 234, 267, 289, 312, 298, 245, 189, 156]
        
        fig = px.line(x=hours, y=api_calls, markers=True)
        fig.update_layout(
            xaxis_title="Hour",
            yaxis_title="API Calls",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity
    st.markdown("### 📋 Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🤖 AI Employee Activity")
        activities = [
            "Research Analyst completed market analysis for AAPL",
            "Trader executed 15 trades with 87% success rate",
            "Risk Manager updated portfolio risk assessment",
            "Compliance Officer reviewed regulatory requirements",
            "Data Specialist processed 2.3M data points"
        ]
        
        for activity in activities:
            st.markdown(f"• {activity}")
    
    with col2:
        st.markdown("#### 🧠 Self-Improvement Activity")
        improvements = [
            "Generated proposal: Optimize RL engine performance",
            "Approved: Enhanced market data processing",
            "Implemented: Improved error handling system",
            "Analyzed: System architecture optimization",
            "Pending: Advanced explainability features"
        ]
        
        for improvement in improvements:
            st.markdown(f"• {improvement}")

def render_employees():
    """Render AI Employee management page."""
    st.markdown("## 👥 AI Employee Management")
    
    # Employee creation section
    with st.expander("➕ Create New AI Employee", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            role_options = [(r["name"], r["key"]) for r in ROLES]
            selected_role = st.selectbox("Role", [r[0] for r in role_options])
            specialization = st.text_input("Specialization", placeholder="e.g., cryptocurrency markets, high-frequency trading")
        
        with col2:
            advanced_features = st.multiselect(
                "Advanced Features",
                ["Real-time Market Data", "Advanced RL", "Explainability", "Self-Learning", "Risk Management"],
                default=["Real-time Market Data", "Advanced RL", "Explainability"]
            )
            
            if st.button("🚀 Create AI Employee", type="primary"):
                if specialization:
                    role_key = dict(role_options)[selected_role]
                    new_employee = {
                        "id": f"emp_{uuid.uuid4().hex[:8]}",
                        "role": role_key,
                        "role_name": selected_role,
                        "specialization": specialization,
                        "features": advanced_features,
                        "status": "initializing",
                        "created_at": datetime.now(),
                        "performance": {
                            "accuracy": 0.0,
                            "success_rate": 0.0,
                            "response_time": 0.0
                        }
                    }
                    st.session_state["employees"].append(new_employee)
                    st.success(f"✅ Created {selected_role} AI Employee: {specialization}")
                    st.rerun()
                else:
                    st.error("Please enter a specialization")
    
    # Employee list
    st.markdown("### 📋 Active AI Employees")
    
    if st.session_state["employees"]:
        for employee in st.session_state["employees"]:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="employee-card">
                        <h4>🤖 {employee['role_name']}</h4>
                        <p><strong>Specialization:</strong> {employee['specialization']}</p>
                        <p><strong>ID:</strong> {employee['id']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Advanced Features:**")
                    for feature in employee.get('features', []):
                        st.markdown(f"• {feature}")
                
                with col3:
                    st.markdown("**Performance:**")
                    perf = employee.get('performance', {})
                    st.markdown(f"Accuracy: {perf.get('accuracy', 0):.1f}%")
                    st.markdown(f"Success: {perf.get('success_rate', 0):.1f}%")
                
                with col4:
                    status_color = {
                        "active": "🟢",
                        "initializing": "🟡", 
                        "error": "🔴",
                        "training": "🟠"
                    }
                    st.markdown(f"**Status:** {status_color.get(employee['status'], '⚪')} {employee['status'].title()}")
                    
                    if st.button(f"⚙️ Configure", key=f"config_{employee['id']}"):
                        st.info(f"Configuration panel for {employee['role_name']}")
                    
                    if st.button(f"📊 Monitor", key=f"monitor_{employee['id']}"):
                        st.info(f"Performance monitoring for {employee['role_name']}")
    else:
        st.info("No AI Employees created yet. Create your first AI Employee above!")

def render_performance():
    """Render performance analytics page."""
    st.markdown("## 📈 Performance Analytics")
    
    # Performance overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Overall Performance")
        metrics = get_system_metrics()
        
        st.metric("Success Rate", f"{metrics.get('success_rate', 0)}%")
        st.metric("Avg Response Time", metrics.get('avg_response_time', 'N/A'))
        st.metric("System Uptime", metrics.get('system_uptime', 'N/A'))
    
    with col2:
        st.markdown("### 📊 AI Employee Performance")
        
        if st.session_state["employees"]:
            # Performance chart
            roles = [emp['role_name'] for emp in st.session_state["employees"]]
            accuracies = [emp.get('performance', {}).get('accuracy', 0) for emp in st.session_state["employees"]]
            
            fig = px.bar(x=roles, y=accuracies, title="Accuracy by Role")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No AI Employees to analyze")
    
    with col3:
        st.markdown("### 💰 Cost Analysis")
        
        # Cost breakdown
        cost_data = {
            'Category': ['API Calls', 'Compute', 'Storage', 'Data Feeds'],
            'Cost': [8.45, 2.10, 1.20, 0.70]
        }
        df_cost = pd.DataFrame(cost_data)
        
        fig = px.pie(df_cost, values='Cost', names='Category', title="Monthly Cost Breakdown")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analytics
    st.markdown("### 📊 Detailed Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🎯 KPIs", "🔍 Insights"])
    
    with tab1:
        st.markdown("#### Performance Trends (Last 30 Days)")
        
        # Simulated trend data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        success_rates = [85 + i * 0.3 + np.random.normal(0, 2) for i in range(30)]
        response_times = [1.5 - i * 0.01 + np.random.normal(0, 0.1) for i in range(30)]
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Success Rate Trend", "Response Time Trend"))
        
        fig.add_trace(go.Scatter(x=dates, y=success_rates, name="Success Rate"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=response_times, name="Response Time"), row=2, col=1)
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Key Performance Indicators")
        
        kpi_data = {
            'Metric': ['Total AI Employees', 'Active Employees', 'Training Success Rate', 'Deployment Success Rate', 'System Reliability'],
            'Value': [len(st.session_state["employees"]), len([e for e in st.session_state["employees"] if e.get('status') == 'active']), '94.2%', '98.7%', '99.8%'],
            'Target': ['50', '45', '95%', '99%', '99.9%'],
            'Status': ['🟢 On Track', '🟢 On Track', '🟡 Below Target', '🟢 On Track', '🟡 Below Target']
        }
        
        df_kpi = pd.DataFrame(kpi_data)
        st.dataframe(df_kpi, use_container_width=True)
    
    with tab3:
        st.markdown("#### AI Insights")
        
        insights = [
            "🎯 **Performance Insight**: Research Analysts show 15% higher accuracy when using real-time market data",
            "⚡ **Optimization Opportunity**: Trader response time can be improved by 23% with RL optimization",
            "📊 **Trend Analysis**: System performance has improved 8.5% over the last 30 days",
            "🔍 **Risk Alert**: 2 AI Employees approaching performance thresholds",
            "💡 **Recommendation**: Consider scaling up Data Specialist capacity for peak market hours"
        ]
        
        for insight in insights:
            st.markdown(f"• {insight}")

def render_self_improvement():
    """Render self-improvement hub."""
    st.markdown("## 🧠 Self-Improvement Hub")
    st.markdown("*True AI Self-Improvement with Human-in-the-Loop Approval*")
    
    # Self-improvement overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Improvement Statistics")
        st.metric("Total Proposals", "23")
        st.metric("Approved Changes", "18")
        st.metric("Success Rate", "78.3%")
    
    with col2:
        st.markdown("### 🎯 Current Status")
        st.metric("Pending Approvals", "3")
        st.metric("In Progress", "2")
        st.metric("Last Improvement", "2 hours ago")
    
    with col3:
        st.markdown("### 💰 Impact Metrics")
        st.metric("Performance Gain", "+12.5%")
        st.metric("Cost Reduction", "-8.2%")
        st.metric("Efficiency Boost", "+15.7%")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Analyze System", type="primary", use_container_width=True):
            st.info("🔍 **System Analysis Started**\n\nAnalyzing current system performance, architecture, and identifying improvement opportunities...")
    
    with col2:
        if st.button("📝 Generate Proposals", type="primary", use_container_width=True):
            st.info("📝 **Proposal Generation Started**\n\nGenerating improvement proposals based on system analysis and performance data...")
    
    with col3:
        if st.button("📊 View History", type="primary", use_container_width=True):
            st.info("📊 **Loading Improvement History**\n\nDisplaying all past improvements, their impact, and learning outcomes...")
    
    # Recent proposals
    st.markdown("### 📋 Recent Improvement Proposals")
    
    proposals = [
        {
            "id": "PROP-001",
            "title": "Optimize RL Engine Performance",
            "description": "Enhance reinforcement learning engine with advanced meta-learning algorithms",
            "impact": "High",
            "status": "pending",
            "created": "2 hours ago"
        },
        {
            "id": "PROP-002", 
            "title": "Improve Market Data Processing",
            "description": "Implement parallel processing for real-time market data feeds",
            "impact": "Medium",
            "status": "approved",
            "created": "1 day ago"
        },
        {
            "id": "PROP-003",
            "title": "Enhanced Error Handling",
            "description": "Add comprehensive error handling and recovery mechanisms",
            "impact": "High", 
            "status": "implemented",
            "created": "3 days ago"
        }
    ]
    
    for proposal in proposals:
        status_class = proposal['status']
        status_icon = {
            'pending': '🟡',
            'approved': '🟢', 
            'rejected': '🔴',
            'implemented': '✅'
        }
        
        st.markdown(f"""
        <div class="proposal-card {status_class}">
            <h4>{status_icon[proposal['status']]} {proposal['title']}</h4>
            <p><strong>ID:</strong> {proposal['id']} | <strong>Impact:</strong> {proposal['impact']} | <strong>Created:</strong> {proposal['created']}</p>
            <p>{proposal['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button(f"✅ Approve", key=f"approve_{proposal['id']}", disabled=proposal['status'] != 'pending'):
                st.success(f"Approved proposal {proposal['id']}")
        with col2:
            if st.button(f"❌ Reject", key=f"reject_{proposal['id']}", disabled=proposal['status'] != 'pending'):
                st.error(f"Rejected proposal {proposal['id']}")
        with col3:
            if st.button(f"📊 Details", key=f"details_{proposal['id']}"):
                st.info(f"Detailed analysis for {proposal['id']}")
        with col4:
            if st.button(f"📝 Edit", key=f"edit_{proposal['id']}", disabled=proposal['status'] != 'pending'):
                st.info(f"Edit proposal {proposal['id']}")

def render_market_data():
    """Render real-time market data page."""
    st.markdown("## 📊 Real-time Market Data")
    st.markdown("*Live financial data feeds and market analysis*")
    
    # Market overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 📈 Market Status")
        st.metric("Active Symbols", "156")
        st.metric("Data Sources", "8")
        st.metric("Update Frequency", "1s")
    
    with col2:
        st.markdown("### 🔄 Data Quality")
        st.metric("Success Rate", "99.2%")
        st.metric("Latency", "45ms")
        st.metric("Accuracy", "99.8%")
    
    with col3:
        st.markdown("### 📊 Volume")
        st.metric("Daily Volume", "2.3B")
        st.metric("Trades/sec", "1,247")
        st.metric("Data Points", "45.2M")
    
    with col4:
        st.markdown("### ⚠️ Alerts")
        st.metric("Active Alerts", "3")
        st.metric("Critical", "1")
        st.metric("Warnings", "2")
    
    # Market data visualization
    st.markdown("### 📈 Live Market Data")
    
    # Simulated market data
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD"]
    prices = [185.23, 142.56, 378.91, 245.67, 43250.00, 2650.00]
    changes = [1.2, -0.8, 2.1, -1.5, 3.2, -0.9]
    
    # Create market data table
    market_data = pd.DataFrame({
        'Symbol': symbols,
        'Price': prices,
        'Change %': changes,
        'Volume': [45.2, 23.1, 67.8, 89.3, 12.4, 8.9]
    })
    
    st.dataframe(market_data, use_container_width=True)
    
    # Price charts
    st.markdown("### 📊 Price Charts")
    
    tab1, tab2, tab3 = st.tabs(["📈 Stock Prices", "📊 Crypto", "📉 Volatility"])
    
    with tab1:
        # Simulated stock price data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        fig = go.Figure()
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            base_price = {"AAPL": 180, "GOOGL": 140, "MSFT": 375}[symbol]
            prices = [base_price + i * 0.5 + np.random.normal(0, 2) for i in range(30)]
            fig.add_trace(go.Scatter(x=dates, y=prices, name=symbol))
        
        fig.update_layout(title="Stock Prices (30 Days)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Crypto price data
        fig = go.Figure()
        for symbol in ["BTC-USD", "ETH-USD"]:
            base_price = {"BTC-USD": 43000, "ETH-USD": 2600}[symbol]
            prices = [base_price + i * 50 + np.random.normal(0, 200) for i in range(30)]
            fig.add_trace(go.Scatter(x=dates, y=prices, name=symbol))
        
        fig.update_layout(title="Cryptocurrency Prices (30 Days)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Volatility chart
        volatility_data = [2.1, 2.3, 1.8, 2.5, 2.0, 2.7, 2.2, 1.9, 2.4, 2.1]
        fig = px.bar(x=list(range(10)), y=volatility_data, title="Market Volatility Index")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_settings():
    """Render system configuration page."""
    st.markdown("## 🔧 System Configuration")
    
    # Configuration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ General", "🔑 API Keys", "🧠 AI Settings", "📊 Monitoring"])
    
    with tab1:
        st.markdown("### General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Theme", ["Dark", "Light", "Auto"])
            st.selectbox("Language", ["English", "Spanish", "French", "German"])
            st.selectbox("Timezone", ["UTC", "EST", "PST", "GMT"])
        
        with col2:
            st.number_input("Max AI Employees", min_value=1, max_value=100, value=50)
            st.number_input("Training Timeout (hours)", min_value=1, max_value=48, value=24)
            st.number_input("Optimization Interval (hours)", min_value=1, max_value=24, value=6)
        
        if st.button("💾 Save General Settings", type="primary"):
            st.success("General settings saved successfully!")
    
    with tab2:
        st.markdown("### API Configuration")
        
        openai_key = st.text_input("OpenAI API Key", type="password", value="sk-...")
        model = st.selectbox("OpenAI Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
        
        st.markdown("### Additional API Keys")
        alpha_vantage_key = st.text_input("Alpha Vantage API Key", type="password")
        quandl_key = st.text_input("Quandl API Key", type="password")
        
        if st.button("💾 Save API Settings", type="primary"):
            st.success("API settings saved successfully!")
    
    with tab3:
        st.markdown("### AI Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Enable Advanced RL", value=True)
            st.checkbox("Enable Real-time Data", value=True)
            st.checkbox("Enable Explainability", value=True)
            st.checkbox("Enable Self-Improvement", value=True)
        
        with col2:
            st.number_input("Max Tokens", min_value=1000, max_value=8000, value=4000)
            st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
            st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        
        if st.button("💾 Save AI Settings", type="primary"):
            st.success("AI settings saved successfully!")
    
    with tab4:
        st.markdown("### Monitoring Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Enable Performance Monitoring", value=True)
            st.checkbox("Enable Cost Tracking", value=True)
            st.checkbox("Enable Error Logging", value=True)
            st.checkbox("Enable Audit Trail", value=True)
        
        with col2:
            st.number_input("Monthly Spend Limit ($)", min_value=1, max_value=1000, value=15)
            st.number_input("Warning Threshold (%)", min_value=50, max_value=100, value=80)
            st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
        
        if st.button("💾 Save Monitoring Settings", type="primary"):
            st.success("Monitoring settings saved successfully!")

def render_help():
    """Render help and documentation page."""
    st.markdown("## 📋 Help & Documentation")
    
    # Help tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Getting Started", "📚 User Guide", "🔧 Troubleshooting", "📞 Support"])
    
    with tab1:
        st.markdown("### 🚀 Quick Start Guide")
        
        st.markdown("""
        #### 1. **System Overview**
        The Bot Builder AI is an advanced AI Employee management system featuring:
        - 🤖 **AI Employee Creation**: Create specialized AI agents for different roles
        - 🧠 **True Self-Improvement**: AI can analyze and improve itself with your approval
        - 📊 **Real-time Market Data**: Live financial data feeds and analysis
        - 📈 **Advanced Analytics**: Comprehensive performance monitoring and insights
        
        #### 2. **First Steps**
        1. **Create Your First AI Employee**: Go to AI Employee Management and create a Research Analyst
        2. **Monitor Performance**: Check the Performance Analytics dashboard
        3. **Explore Self-Improvement**: Visit the Self-Improvement Hub to see AI proposals
        4. **Configure Settings**: Set up your API keys and preferences
        
        #### 3. **Key Features**
        - **Executive Dashboard**: Overview of all system metrics and performance
        - **AI Employee Management**: Create, configure, and monitor AI agents
        - **Performance Analytics**: Detailed performance insights and trends
        - **Self-Improvement Hub**: AI-generated improvement proposals
        - **Real-time Market Data**: Live financial data and market analysis
        """)
    
    with tab2:
        st.markdown("### 📚 User Guide")
        
        st.markdown("""
        #### **AI Employee Roles**
        
        **🤖 Research Analyst**
        - Deep learning models for market prediction
        - Sentiment analysis of financial news
        - Technical and fundamental analysis
        - Economic indicator analysis
        
        **⚡ Trader**
        - Reinforcement learning for trading decisions
        - High-frequency trading capabilities
        - Portfolio optimization
        - Risk management integration
        
        **🛡️ Risk Manager**
        - Probability theory and statistical modeling
        - Scenario testing and stress analysis
        - Portfolio risk assessment
        - Compliance monitoring
        
        **📋 Compliance Officer**
        - Regulatory knowledge and NLP
        - Policy enforcement and monitoring
        - Audit trail management
        - Explainability features
        
        **📊 Data Specialist**
        - Data cleaning and management
        - Real-time data processing
        - Market data analysis
        - Insights generation
        """)
    
    with tab3:
        st.markdown("### 🔧 Troubleshooting")
        
        st.markdown("""
        #### **Common Issues**
        
        **❌ API Key Errors**
        - Ensure your OpenAI API key is valid and has sufficient credits
        - Check API key permissions and rate limits
        - Verify the key is correctly entered in Settings
        
        **⚠️ Performance Issues**
        - Monitor system resources (CPU, memory)
        - Check network connectivity for real-time data
        - Review error logs for specific issues
        
        **🔴 AI Employee Failures**
        - Verify training data availability
        - Check model configuration settings
        - Review performance metrics for bottlenecks
        
        **📊 Data Issues**
        - Ensure data sources are accessible
        - Check API rate limits for external data
        - Verify data format and quality
        """)
    
    with tab4:
        st.markdown("### 📞 Support")
        
        st.markdown("""
        #### **Getting Help**
        
        **📧 Email Support**
        - Technical issues: tech-support@botbuilder.ai
        - General questions: help@botbuilder.ai
        
        **📚 Documentation**
        - Full documentation: docs.botbuilder.ai
        - API reference: api.botbuilder.ai
        - Tutorials: tutorials.botbuilder.ai
        
        **🐛 Bug Reports**
        - GitHub Issues: github.com/botbuilder/issues
        - Include system logs and error messages
        
        **💡 Feature Requests**
        - Submit via GitHub Discussions
        - Include use case and expected benefits
        """)

def render_chat_interface():
    """Render the AI chat interface."""
    st.markdown("### 💬 AI Assistant")
    
    # Chat messages
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if st.session_state["chat_messages"]:
            for message in st.session_state["chat_messages"]:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="ai-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ai-message">
            👋 Welcome to Bot Builder AI! I'm your AI assistant. How can I help you today?
            
            Try asking:
            • "Create a new Research Analyst AI Employee"
            • "Show me the system performance"
            • "Analyze the system for improvements"
            • "What's the current market status?"
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Message", placeholder="Ask me anything about AI Employees, performance, or system management...", label_visibility="collapsed")
        with col2:
            submitted = st.form_submit_button("Send", use_container_width=True)
    
    # Handle message sending
    if submitted and user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        
        # Get AI response
        ai_response = "I'm processing your request..."
        try:
            session_id = st.session_state.get("chat_session_id")
            if not session_id:
                session_id = str(uuid.uuid4())
                st.session_state["chat_session_id"] = session_id
            
            user_id = st.session_state.get("chat_user_id")
            if not user_id:
                user_id = f"user_{uuid.uuid4().hex[:8]}"
                st.session_state["chat_user_id"] = user_id
            
            # Process with AI engine
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ai_response = loop.run_until_complete(
                st.session_state["ai_engine"].process_user_input(user_input, session_id, user_id)
            )
            loop.close()
            
        except Exception as e:
            ai_response = f"I encountered an error: {str(e)}"
        
        st.session_state["chat_messages"].append({"role": "assistant", "content": ai_response})
        st.rerun()

def main():
    """Main application function."""
    # Render header
    render_header()
    
    # Create layout
    col1, col2, col3 = st.columns([1, 3, 1])
    
    # Left sidebar (navigation)
    with col1:
        render_sidebar()
    
    # Main content
    with col2:
        # Route to appropriate page
        current_page = st.session_state.get("current_page", "dashboard")
        
        if current_page == "dashboard":
            render_dashboard()
        elif current_page == "employees":
            render_employees()
        elif current_page == "performance":
            render_performance()
        elif current_page == "self_improvement":
            render_self_improvement()
        elif current_page == "market_data":
            render_market_data()
        elif current_page == "settings":
            render_settings()
        elif current_page == "help":
            render_help()
        else:
            st.markdown("## Page not found")
    
    # Right sidebar (AI chat)
    with col3:
        render_chat_interface()

if __name__ == "__main__":
    main() 