"""
CEO Streamlit Portal - Executive Command Center
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Import our core modules
from core.ceo_portal import CEOPortal, DecisionCategory, Priority, CEODecision
from core.sdlc_bot_team import SDLCBotTeam, TaskPriority, TaskStatus, BotRole
from core.ai_engine import AIEngine
from data.real_time_market_data import RealTimeMarketDataProvider

# Configure Streamlit page
st.set_page_config(
    page_title="CEO Command Center - Bot Builder AI",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.priority-critical {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    color: white;
    padding: 0.5rem;
    border-radius: 5px;
    font-weight: bold;
}

.priority-high {
    background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
    color: #2c2c2c;
    padding: 0.5rem;
    border-radius: 5px;
    font-weight: bold;
}

.priority-medium {
    background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
    color: white;
    padding: 0.5rem;
    border-radius: 5px;
    font-weight: bold;
}

.team-status-healthy {
    background: linear-gradient(135deg, #00d2d3 0%, #54a0ff 100%);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem;
}

.team-status-attention {
    background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem;
}

.decision-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.quick-action-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.8rem 1.5rem;
    border-radius: 25px;
    font-weight: bold;
    margin: 0.2rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.executive-summary {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ceo_portal' not in st.session_state:
    st.session_state.ceo_portal = None
if 'sdlc_team' not in st.session_state:
    st.session_state.sdlc_team = None
if 'ai_engine' not in st.session_state:
    st.session_state.ai_engine = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

async def initialize_systems():
    """Initialize all CEO portal systems."""
    if st.session_state.ceo_portal is None:
        with st.spinner("🚀 Initializing CEO Command Center..."):
            # Initialize CEO Portal
            st.session_state.ceo_portal = CEOPortal()
            await st.session_state.ceo_portal.initialize()
            
            # Initialize SDLC Bot Team
            st.session_state.sdlc_team = SDLCBotTeam(st.session_state.ceo_portal)
            await st.session_state.sdlc_team.initialize()
            
            # Initialize AI Engine
            st.session_state.ai_engine = AIEngine()
            await st.session_state.ai_engine.initialize()
            
            st.success("✅ CEO Command Center initialized successfully!")

def create_executive_summary_card(summary: Dict[str, Any]):
    """Create executive summary card."""
    st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Pending Decisions", 
            summary.get('total_pending_decisions', 0),
            delta=f"🔴 {summary.get('critical_decisions', 0)} Critical"
        )
    
    with col2:
        financial_impact = summary.get('financial_summary', {}).get('total_impact', 0)
        st.metric(
            "Financial Impact", 
            f"${financial_impact:,.0f}",
            delta="Pending Decisions"
        )
    
    with col3:
        risk_level = summary.get('risk_summary', {}).get('average_risk', 0)
        st.metric(
            "Risk Level", 
            f"{risk_level:.1%}",
            delta="Average Risk"
        )
    
    with col4:
        strategic_progress = summary.get('strategic_progress', {}).get('overall_alignment', 0)
        st.metric(
            "Strategic Alignment", 
            f"{strategic_progress:.1%}",
            delta="Overall Progress"
        )
    
    # Key insights
    if summary.get('key_insights'):
        st.subheader("🎯 Key Insights")
        for insight in summary['key_insights'][:3]:
            st.info(f"💡 {insight}")
    
    # Recommendations
    if summary.get('recommendations'):
        st.subheader("📋 Recommendations")
        for rec in summary['recommendations'][:2]:
            st.warning(f"⚠️ {rec}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_decision_queue_section(dashboard: Dict[str, Any]):
    """Create decision queue section."""
    st.header("🎯 Executive Decision Queue")
    
    pending_decisions = dashboard.get('pending_decisions', {})
    
    # Critical decisions
    critical_decisions = pending_decisions.get('critical', [])
    if critical_decisions:
        st.subheader("🚨 Critical Decisions (Immediate Attention Required)")
        
        for decision in critical_decisions:
            with st.expander(f"🔴 {decision['title']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {decision['description']}")
                    st.write(f"**Requesting Bot:** {decision['requesting_bot']}")
                    st.write(f"**Team:** {decision['requesting_team']}")
                    st.write(f"**Financial Impact:** ${decision['financial_impact']:,.0f}")
                    st.write(f"**Risk Level:** {decision['risk_level']:.1%}")
                    st.write(f"**Deadline:** {decision['deadline']}")
                
                with col2:
                    st.markdown('<div class="priority-critical">CRITICAL</div>', unsafe_allow_html=True)
                    
                    if st.button(f"✅ Approve", key=f"approve_{decision['id']}"):
                        await handle_ceo_decision(decision['id'], "APPROVED", "Approved by CEO")
                        st.rerun()
                    
                    if st.button(f"❌ Reject", key=f"reject_{decision['id']}"):
                        await handle_ceo_decision(decision['id'], "REJECTED", "Rejected by CEO")
                        st.rerun()
                    
                    if st.button(f"📋 Details", key=f"details_{decision['id']}"):
                        st.session_state.selected_decision = decision['id']
    
    # High priority decisions
    high_decisions = pending_decisions.get('high', [])
    if high_decisions:
        st.subheader("🟡 High Priority Decisions")
        
        for decision in high_decisions[:5]:  # Show top 5
            with st.expander(f"🟡 {decision['title']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Description:** {decision['description'][:200]}...")
                    st.write(f"**Financial Impact:** ${decision['financial_impact']:,.0f}")
                    st.write(f"**Escalation Reason:** {decision['escalation_reason']}")
                
                with col2:
                    st.markdown('<div class="priority-high">HIGH</div>', unsafe_allow_html=True)
                    
                    col2a, col2b = st.columns(2)
                    with col2a:
                        if st.button("✅", key=f"approve_h_{decision['id']}"):
                            await handle_ceo_decision(decision['id'], "APPROVED", "Approved")
                            st.rerun()
                    with col2b:
                        if st.button("❌", key=f"reject_h_{decision['id']}"):
                            await handle_ceo_decision(decision['id'], "REJECTED", "Rejected")
                            st.rerun()

def create_team_status_section(dashboard: Dict[str, Any]):
    """Create team status monitoring section."""
    st.header("👥 Team Status Overview")
    
    team_status = dashboard.get('team_status', {})
    
    if team_status:
        # Team health overview
        teams_df = pd.DataFrame([
            {
                'Team': team_name,
                'Health Score': status['health_score'],
                'Active Bots': status['active_bots'],
                'Total Tasks': status['total_tasks'],
                'Completed': status['completed_tasks'],
                'Blocked': status['blocked_tasks'],
                'Type': status['team_type']
            }
            for team_name, status in team_status.items()
        ])
        
        # Health score chart
        fig = px.bar(
            teams_df, 
            x='Team', 
            y='Health Score',
            color='Health Score',
            color_continuous_scale='RdYlGn',
            title="Team Health Scores"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Team details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 High Performing Teams")
            healthy_teams = teams_df[teams_df['Health Score'] > 0.8].sort_values('Health Score', ascending=False)
            
            for _, team in healthy_teams.iterrows():
                st.markdown(f"""
                <div class="team-status-healthy">
                    <strong>{team['Team']}</strong><br>
                    Health: {team['Health Score']:.1%} | 
                    Tasks: {team['Completed']}/{team['Total Tasks']} | 
                    Bots: {team['Active Bots']}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("⚠️ Teams Needing Attention")
            attention_teams = teams_df[teams_df['Health Score'] <= 0.8].sort_values('Health Score')
            
            for _, team in attention_teams.iterrows():
                st.markdown(f"""
                <div class="team-status-attention">
                    <strong>{team['Team']}</strong><br>
                    Health: {team['Health Score']:.1%} | 
                    Blocked: {team['Blocked']} | 
                    Issues: Low Performance
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.info("No team data available. Teams are initializing...")

def create_real_time_market_section():
    """Create real-time market data section."""
    st.header("📈 Real-Time Market Intelligence")
    
    if st.session_state.ai_engine and st.session_state.ai_engine.real_time_enabled:
        # Get market summary
        market_summary = asyncio.run(st.session_state.ai_engine.get_real_time_market_summary())
        
        if market_summary and 'symbols' in market_summary:
            # Market overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Active Symbols", len(market_summary['symbols']))
            with col2:
                st.metric("Active Connections", market_summary.get('active_connections', 0))
            with col3:
                st.metric("Data Quality", f"{market_summary.get('data_quality', {}).get('average', 0):.1%}")
            with col4:
                st.metric("Recent Events", market_summary.get('recent_events', 0))
            
            # Symbol performance
            symbols_data = []
            for symbol, data in market_summary['symbols'].items():
                symbols_data.append({
                    'Symbol': symbol,
                    'Price': data['price'],
                    'Change %': data['change_percent'],
                    'Volume': data['volume'],
                    'Last Update': data['last_update']
                })
            
            if symbols_data:
                symbols_df = pd.DataFrame(symbols_data)
                
                # Price change chart
                fig = px.bar(
                    symbols_df,
                    x='Symbol',
                    y='Change %',
                    color='Change %',
                    color_continuous_scale='RdYlGn',
                    title="Market Performance Today"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Market data table
                st.subheader("📊 Live Market Data")
                st.dataframe(
                    symbols_df,
                    use_container_width=True,
                    column_config={
                        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                        "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
                        "Volume": st.column_config.NumberColumn("Volume", format="%d")
                    }
                )
    else:
        st.info("Real-time market data is initializing...")

def create_quick_actions_section(dashboard: Dict[str, Any]):
    """Create quick actions section."""
    st.header("⚡ Quick Actions")
    
    quick_actions = dashboard.get('quick_actions', [])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Review All Critical Decisions", use_container_width=True):
            st.session_state.view_mode = "critical_decisions"
            st.rerun()
    
    with col2:
        if st.button("👥 Check Team Performance", use_container_width=True):
            st.session_state.view_mode = "team_performance"
            st.rerun()
    
    with col3:
        if st.button("📈 Market Analysis", use_container_width=True):
            st.session_state.view_mode = "market_analysis"
            st.rerun()
    
    # Display suggested actions
    if quick_actions:
        st.subheader("🎯 Suggested Actions")
        
        for action in quick_actions:
            urgency_color = {
                "HIGH": "🔴",
                "MEDIUM": "🟡", 
                "LOW": "🟢"
            }.get(action.get('urgency', 'LOW'), '🟢')
            
            st.info(f"{urgency_color} **{action['title']}** - {action['description']}")

def create_system_health_section(dashboard: Dict[str, Any]):
    """Create system health monitoring section."""
    st.header("🏥 System Health")
    
    system_health = dashboard.get('system_health', {})
    
    if system_health:
        # Overall health gauge
        overall_health = system_health.get('overall_health', 0.5)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_health * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall System Health"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Health metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Team Health", 
                f"{system_health.get('team_health', 0.5):.1%}",
                delta="Average across teams"
            )
        
        with col2:
            st.metric(
                "Decision Queue", 
                f"{system_health.get('decision_queue_health', 0.5):.1%}",
                delta="Processing efficiency"
            )
        
        with col3:
            st.metric(
                "Active Teams", 
                system_health.get('active_teams', 0),
                delta="Currently operational"
            )
        
        # Status indicator
        status = system_health.get('status', 'UNKNOWN')
        status_colors = {
            'HEALTHY': '🟢',
            'ATTENTION_NEEDED': '🟡',
            'CRITICAL': '🔴',
            'UNKNOWN': '⚪'
        }
        
        st.info(f"{status_colors.get(status, '⚪')} **System Status:** {status}")

async def handle_ceo_decision(decision_id: str, action: str, response: str):
    """Handle CEO decision approval/rejection."""
    if st.session_state.ceo_portal:
        approved = action == "APPROVED"
        result = await st.session_state.ceo_portal.process_ceo_decision(
            decision_id, 
            response, 
            approved
        )
        
        if result['success']:
            st.success(f"✅ Decision {action.lower()} successfully!")
        else:
            st.error(f"❌ Error processing decision: {result.get('error', 'Unknown error')}")

def create_sidebar():
    """Create sidebar with navigation and controls."""
    st.sidebar.markdown('<div class="main-header">CEO Portal</div>', unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.subheader("🧭 Navigation")
    view_mode = st.sidebar.radio(
        "Select View",
        ["Executive Dashboard", "Decision Queue", "Team Management", "Market Intelligence", "System Health"],
        key="nav_radio"
    )
    
    # Quick stats
    st.sidebar.subheader("📊 Quick Stats")
    if st.session_state.ceo_portal:
        dashboard = asyncio.run(st.session_state.ceo_portal.get_ceo_dashboard())
        
        pending_count = dashboard.get('pending_decisions', {}).get('total_count', 0)
        critical_count = len(dashboard.get('pending_decisions', {}).get('critical', []))
        
        st.sidebar.metric("Pending Decisions", pending_count)
        st.sidebar.metric("Critical Items", critical_count)
        
        # System status
        system_health = dashboard.get('system_health', {})
        status = system_health.get('status', 'UNKNOWN')
        health_score = system_health.get('overall_health', 0.5)
        
        st.sidebar.metric("System Health", f"{health_score:.1%}")
        st.sidebar.info(f"Status: {status}")
    
    # Controls
    st.sidebar.subheader("⚙️ Controls")
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Emergency controls
    st.sidebar.subheader("🚨 Emergency Controls")
    if st.sidebar.button("🛑 Emergency Stop All Bots"):
        st.sidebar.error("Emergency stop activated!")
    
    if st.sidebar.button("📞 Escalate to CEO"):
        st.sidebar.warning("Manual escalation requested!")
    
    return view_mode

def main():
    """Main CEO Portal application."""
    # Header
    st.markdown('<div class="main-header">👔 CEO Command Center</div>', unsafe_allow_html=True)
    st.markdown("**Strategic oversight for your AI organization**")
    
    # Initialize systems
    asyncio.run(initialize_systems())
    
    # Sidebar navigation
    view_mode = create_sidebar()
    
    # Get dashboard data
    dashboard = {}
    if st.session_state.ceo_portal:
        dashboard = asyncio.run(st.session_state.ceo_portal.get_ceo_dashboard())
    
    # Main content based on view mode
    if view_mode == "Executive Dashboard":
        # Executive summary
        if dashboard.get('executive_summary'):
            create_executive_summary_card(dashboard['executive_summary'])
        
        # Quick overview sections
        col1, col2 = st.columns(2)
        
        with col1:
            # Critical decisions preview
            critical_decisions = dashboard.get('pending_decisions', {}).get('critical', [])
            if critical_decisions:
                st.subheader("🚨 Critical Decisions")
                for decision in critical_decisions[:3]:
                    st.error(f"🔴 {decision['title']} - ${decision['financial_impact']:,.0f}")
            
            # Quick actions
            create_quick_actions_section(dashboard)
        
        with col2:
            # Team status summary
            team_status = dashboard.get('team_status', {})
            if team_status:
                st.subheader("👥 Team Summary")
                
                healthy_teams = len([t for t in team_status.values() if t['health_score'] > 0.8])
                total_teams = len(team_status)
                
                st.metric("Healthy Teams", f"{healthy_teams}/{total_teams}")
                
                # Top performing team
                if team_status:
                    top_team = max(team_status.items(), key=lambda x: x[1]['health_score'])
                    st.success(f"🏆 Top Team: {top_team[0]} ({top_team[1]['health_score']:.1%})")
            
            # System health summary
            system_health = dashboard.get('system_health', {})
            if system_health:
                st.subheader("🏥 System Health")
                overall_health = system_health.get('overall_health', 0.5)
                status = system_health.get('status', 'UNKNOWN')
                
                st.metric("Overall Health", f"{overall_health:.1%}")
                st.info(f"Status: {status}")
    
    elif view_mode == "Decision Queue":
        create_decision_queue_section(dashboard)
    
    elif view_mode == "Team Management":
        create_team_status_section(dashboard)
    
    elif view_mode == "Market Intelligence":
        create_real_time_market_section()
    
    elif view_mode == "System Health":
        create_system_health_section(dashboard)
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated:** {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("*CEO Command Center - Strategic AI Organization Management*")

if __name__ == "__main__":
    main() 