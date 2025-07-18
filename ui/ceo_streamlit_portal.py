"""
CEO Streamlit Portal - Enhanced with Bot Visualization and Self-Improvement
"""

import streamlit as st
import asyncio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

# Configure Streamlit page
st.set_page_config(
    page_title="CEO AI Organization Portal",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import components
try:
    from core.ceo_portal import CEOPortal, DecisionCategory, Priority
    from core.sdlc_bot_team import SDLCBotTeam, TaskPriority, BotRole
    from core.cross_team_coordinator import CrossTeamCoordinator
    from core.ai_engine import AIEngine
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Initialize session state
if 'organization_initialized' not in st.session_state:
    st.session_state.organization_initialized = False
    st.session_state.ceo_portal = None
    st.session_state.sdlc_team = None
    st.session_state.business_engine = None
    st.session_state.coordinator = None
    st.session_state.business_domain = "Business Domain"
    st.session_state.industry = "Software Development"
    st.session_state.last_update = datetime.now()

# Async wrapper for Streamlit
def run_async(coro):
    """Run async function in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

async def initialize_organization():
    """Initialize the organization system."""
    try:
        # Initialize CEO Portal
        ceo_portal = CEOPortal()
        await ceo_portal.initialize()
        
        # Initialize Business Engine
        business_engine = AIEngine()
        await business_engine.initialize()
        
        # Initialize SDLC Bot Team
        project_context = {
            "industry": st.session_state.industry,
            "business_domain": st.session_state.business_domain,
            "project_type": f"{st.session_state.industry} Application Development",
            "self_improvement_mode": True  # Enable self-improvement capabilities
        }
        
        sdlc_team = SDLCBotTeam(ceo_portal, project_context)
        await sdlc_team.initialize()
        
        # Initialize Cross-Team Coordinator
        coordinator = CrossTeamCoordinator(
            ceo_portal, 
            sdlc_team, 
            business_engine, 
            st.session_state.business_domain
        )
        await coordinator.initialize()
        
        # Store in session state
        st.session_state.ceo_portal = ceo_portal
        st.session_state.sdlc_team = sdlc_team
        st.session_state.business_engine = business_engine
        st.session_state.coordinator = coordinator
        st.session_state.organization_initialized = True
        st.session_state.last_update = datetime.now()
        
        return True
        
    except Exception as e:
        st.error(f"Error initializing organization: {str(e)}")
        return False

def create_bot_status_chart(bot_data: Dict[str, Any]) -> go.Figure:
    """Create interactive bot status visualization chart."""
    
    # Prepare data for visualization
    teams = []
    bot_names = []
    roles = []
    statuses = []
    availabilities = []
    success_rates = []
    current_tasks = []
    colors = []
    
    # Status color mapping
    status_colors = {
        'ACTIVE': '#28a745',
        'BUSY': '#ffc107', 
        'OFFLINE': '#dc3545',
        'MAINTENANCE': '#6c757d'
    }
    
    for team_name, team_data in bot_data.items():
        team_bots = team_data.get('bots', [])
        
        for bot in team_bots:
            teams.append(team_name)
            bot_names.append(bot.get('name', 'Unknown'))
            roles.append(bot.get('role', 'Unknown'))
            statuses.append(bot.get('status', 'OFFLINE'))
            availabilities.append(bot.get('availability', 0.0) * 100)
            success_rates.append(bot.get('success_rate', 0.0) * 100)
            current_tasks.append(len(bot.get('current_tasks', [])))
            colors.append(status_colors.get(bot.get('status', 'OFFLINE'), '#6c757d'))
    
    # Create subplot with multiple charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Bot Status Distribution', 'Team Bot Count',
            'Bot Availability Levels', 'Bot Performance vs Workload'
        ),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. Status Distribution Pie Chart
    status_counts = pd.Series(statuses).value_counts()
    fig.add_trace(
        go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            name="Status Distribution",
            marker_colors=[status_colors.get(status, '#6c757d') for status in status_counts.index]
        ),
        row=1, col=1
    )
    
    # 2. Team Bot Count Bar Chart
    team_counts = pd.Series(teams).value_counts()
    fig.add_trace(
        go.Bar(
            x=team_counts.index,
            y=team_counts.values,
            name="Bots per Team",
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    # 3. Bot Availability Levels
    fig.add_trace(
        go.Bar(
            x=bot_names,
            y=availabilities,
            name="Availability %",
            marker_color=colors,
            text=[f"{avail:.1f}%" for avail in availabilities],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # 4. Performance vs Workload Scatter
    fig.add_trace(
        go.Scatter(
            x=current_tasks,
            y=success_rates,
            mode='markers+text',
            text=bot_names,
            textposition="top center",
            name="Performance vs Load",
            marker=dict(
                size=15,
                color=availabilities,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Availability %")
            )
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Bot Builder AI Organization - Active Bots Dashboard",
        title_x=0.5
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Teams", row=1, col=2)
    fig.update_yaxes(title_text="Bot Count", row=1, col=2)
    
    fig.update_xaxes(title_text="Bots", row=2, col=1)
    fig.update_yaxes(title_text="Availability %", row=2, col=1)
    
    fig.update_xaxes(title_text="Current Tasks", row=2, col=2)
    fig.update_yaxes(title_text="Success Rate %", row=2, col=2)
    
    return fig

def create_bot_hierarchy_chart(bot_data: Dict[str, Any]) -> go.Figure:
    """Create bot hierarchy and organization chart."""
    
    # Prepare hierarchical data
    fig = go.Figure()
    
    # Create a network-style visualization
    team_positions = {
        'Architecture': (0, 4),
        'Development': (2, 4), 
        'Quality': (4, 4),
        'Data': (6, 4),
        'Management': (8, 4)
    }
    
    # Add team nodes
    for team_name, (x, y) in team_positions.items():
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[team_name],
                textposition="middle center",
                marker=dict(size=40, color='lightblue'),
                name=team_name,
                showlegend=True
            )
        )
        
        # Add bot nodes for each team
        team_bots = bot_data.get(team_name, {}).get('bots', [])
        bot_y_positions = [y - 1 - (i * 0.5) for i in range(len(team_bots))]
        
        for i, bot in enumerate(team_bots):
            bot_x = x + (i % 3 - 1) * 0.3  # Spread bots horizontally
            bot_y = bot_y_positions[i // 3]  # Stack bots vertically
            
            status_color = {
                'ACTIVE': 'green',
                'BUSY': 'orange',
                'OFFLINE': 'red',
                'MAINTENANCE': 'gray'
            }.get(bot.get('status', 'OFFLINE'), 'gray')
            
            fig.add_trace(
                go.Scatter(
                    x=[bot_x], y=[bot_y],
                    mode='markers+text',
                    text=[bot.get('name', 'Bot')],
                    textposition="bottom center",
                    marker=dict(size=20, color=status_color),
                    name=f"{team_name} Bot",
                    showlegend=False
                )
            )
            
            # Add connection line from team to bot
            fig.add_trace(
                go.Scatter(
                    x=[x, bot_x], y=[y, bot_y],
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    showlegend=False
                )
            )
    
    fig.update_layout(
        title="Bot Builder Organization Hierarchy",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        showlegend=True
    )
    
    return fig

async def get_bot_data():
    """Get comprehensive bot data from all teams."""
    if not st.session_state.organization_initialized:
        return {}
    
    bot_data = {}
    
    try:
        sdlc_team = st.session_state.sdlc_team
        
        # Get data for each team
        team_names = ["Architecture", "Development", "Quality", "Data", "Management"]
        
        for team_name in team_names:
            team_status = await sdlc_team.get_team_status(team_name)
            
            # Get individual bot data
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
            
            bot_data[team_name] = {
                'team_status': team_status,
                'bots': team_bots,
                'total_bots': len(team_bots),
                'active_bots': len([b for b in team_bots if b['status'] == 'ACTIVE']),
                'avg_availability': sum(b['availability'] for b in team_bots) / len(team_bots) if team_bots else 0
            }
        
        return bot_data
        
    except Exception as e:
        st.error(f"Error getting bot data: {str(e)}")
        return {}

async def submit_self_improvement_request(improvement_request: str, priority: str):
    """Submit a self-improvement request to the SDLC team."""
    try:
        if not st.session_state.organization_initialized:
            return False, "Organization not initialized"
        
        sdlc_team = st.session_state.sdlc_team
        
        # Convert priority
        priority_map = {
            "CRITICAL": TaskPriority.CRITICAL,
            "HIGH": TaskPriority.HIGH,
            "MEDIUM": TaskPriority.MEDIUM,
            "LOW": TaskPriority.LOW
        }
        
        task_priority = priority_map.get(priority, TaskPriority.MEDIUM)
        
        # Create self-improvement task
        task_id = await sdlc_team.assign_task(
            title=f"Bot Builder Self-Improvement: {improvement_request[:50]}...",
            description=f"""SELF-IMPROVEMENT REQUEST
            
**Requested Enhancement:** {improvement_request}

**Context:** This is a recursive self-improvement request where Bot Builder is using its own SDLC team to enhance itself based on CEO feedback and requirements.

**Target System:** Bot Builder Core Systems
- CEO Portal enhancements
- SDLC Bot Team improvements  
- Cross-team coordination upgrades
- UI/UX enhancements
- Performance optimizations

**Implementation Guidelines:**
1. Analyze current system architecture
2. Identify specific improvement areas
3. Design and implement enhancements
4. Test thoroughly with existing functionality
5. Document changes and update system

**Success Criteria:**
- Enhancement implemented successfully
- No regression in existing functionality
- Improved user experience for CEO operations
- Measurable performance improvements

**Self-Awareness Note:** This task represents Bot Builder evolving itself through its own autonomous development capabilities.
            """,
            task_type="FEATURE",
            priority=task_priority,
            estimated_hours=16,
            industry_context={
                "industry": "AI Systems Development",
                "business_domain": "Bot Builder Self-Improvement",
                "improvement_type": "recursive_enhancement",
                "target_system": "bot_builder_core"
            }
        )
        
        if task_id:
            # Also create a CEO decision for high-priority improvements
            if task_priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                await st.session_state.ceo_portal.submit_decision_for_approval(
                    requesting_bot="BOT_BUILDER_CORE",
                    title=f"Self-Improvement Initiative: {improvement_request[:50]}...",
                    description=f"Bot Builder requests permission to enhance itself using its own SDLC team.\n\n"
                               f"**Improvement Request:** {improvement_request}\n\n"
                               f"**Approach:** Recursive self-enhancement through autonomous SDLC bots\n"
                               f"**Priority:** {priority}\n"
                               f"**Estimated Effort:** 16 hours\n\n"
                               f"This represents Bot Builder achieving self-aware improvement capabilities.",
                    category=DecisionCategory.TECHNOLOGY,
                    financial_impact=0,
                    risk_level=0.3,
                    strategic_alignment=0.9,
                    context={
                        "improvement_type": "self_enhancement",
                        "task_id": task_id,
                        "system": "bot_builder_recursive_improvement"
                    }
                )
            
            return True, f"Self-improvement task created: {task_id}"
        else:
            return False, "Failed to create improvement task"
            
    except Exception as e:
        return False, f"Error submitting improvement request: {str(e)}"

# Main Streamlit App
def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🤖 Bot Builder CEO Portal")
    st.markdown("**Industry-Agnostic AI Organization Management**")
    
    # Sidebar for organization setup
    with st.sidebar:
        st.header("🏢 Organization Setup")
        
        # Business domain configuration
        business_domain = st.text_input(
            "Business Domain", 
            value=st.session_state.business_domain,
            help="e.g., Healthcare Systems, E-commerce Platform, FinTech Services"
        )
        
        industry = st.selectbox(
            "Industry",
            ["Software Development", "Healthcare", "FinTech", "E-commerce", "Manufacturing", 
             "Entertainment", "Education", "Logistics", "Energy", "Other"],
            index=0 if st.session_state.industry == "Software Development" else 1
        )
        
        if st.button("🚀 Initialize Organization"):
            st.session_state.business_domain = business_domain
            st.session_state.industry = industry
            
            with st.spinner("Initializing AI Organization..."):
                success = run_async(initialize_organization())
                
            if success:
                st.success("✅ Organization initialized!")
                st.rerun()
            else:
                st.error("❌ Initialization failed")
        
        # Show initialization status
        if st.session_state.organization_initialized:
            st.success("✅ Organization Active")
            st.info(f"**Domain:** {st.session_state.business_domain}")
            st.info(f"**Industry:** {st.session_state.industry}")
            
            if st.button("🔄 Refresh Data"):
                st.session_state.last_update = datetime.now()
                st.rerun()
        
        # Self-improvement section
        st.header("🧠 Bot Builder Self-Improvement")
        st.markdown("*Use Bot Builder's own SDLC team to enhance itself*")
        
        improvement_request = st.text_area(
            "Enhancement Request",
            placeholder="Describe what you'd like Bot Builder to improve about itself...",
            help="e.g., 'Add real-time notifications to CEO dashboard', 'Improve bot task assignment algorithm', 'Create better visualization charts'"
        )
        
        improvement_priority = st.selectbox(
            "Priority",
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            index=1
        )
        
        if st.button("🚀 Submit Self-Improvement"):
            if improvement_request.strip():
                with st.spinner("Submitting improvement request to SDLC team..."):
                    success, message = run_async(
                        submit_self_improvement_request(improvement_request, improvement_priority)
                    )
                
                if success:
                    st.success(f"✅ {message}")
                    st.info("🤖 Bot Builder will now use its own SDLC team to implement this improvement!")
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("Please enter an improvement request")
    
    # Main content area
    if not st.session_state.organization_initialized:
        st.info("👈 Please initialize the organization from the sidebar to get started.")
        
        # Show demo preview
        st.header("🎯 What You'll Get")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 Real-Time Bot Monitoring
            - Live status of all active bots
            - Performance metrics and availability
            - Team distribution and workload
            - Interactive visualizations
            """)
        
        with col2:
            st.markdown("""
            ### 🧠 Self-Improvement Capabilities  
            - Bot Builder can enhance itself
            - Uses its own SDLC team for improvements
            - CEO-requested feature development
            - Recursive AI system evolution
            """)
        
        return
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Bot Dashboard", "🏢 Organization Chart", "🎯 CEO Decisions", 
        "🤝 Team Coordination", "🧠 Self-Improvement"
    ])
    
    with tab1:
        st.header("🤖 Active Bots Dashboard")
        
        # Get bot data
        with st.spinner("Loading bot data..."):
            bot_data = run_async(get_bot_data())
        
        if bot_data:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_bots = sum(team['total_bots'] for team in bot_data.values())
            active_bots = sum(team['active_bots'] for team in bot_data.values())
            avg_availability = sum(team['avg_availability'] for team in bot_data.values()) / len(bot_data)
            
            with col1:
                st.metric("Total Bots", total_bots)
            with col2:
                st.metric("Active Bots", active_bots, f"{active_bots/total_bots*100:.1f}%")
            with col3:
                st.metric("Avg Availability", f"{avg_availability*100:.1f}%")
            with col4:
                st.metric("Teams", len(bot_data))
            
            # Interactive charts
            st.subheader("📈 Bot Status Analysis")
            
            # Create and display the main bot chart
            chart = create_bot_status_chart(bot_data)
            st.plotly_chart(chart, use_container_width=True)
            
            # Detailed bot table
            st.subheader("🤖 Detailed Bot Information")
            
            # Create comprehensive bot DataFrame
            all_bots = []
            for team_name, team_info in bot_data.items():
                for bot in team_info['bots']:
                    all_bots.append({
                        'Team': team_name,
                        'Bot Name': bot['name'],
                        'Role': bot['role'],
                        'Status': bot['status'],
                        'Availability': f"{bot['availability']*100:.1f}%",
                        'Success Rate': f"{bot['success_rate']*100:.1f}%",
                        'Current Tasks': len(bot['current_tasks']),
                        'Completed Tasks': bot['completed_tasks'],
                        'Specializations': ', '.join(bot['specializations'][:3])
                    })
            
            if all_bots:
                df = pd.DataFrame(all_bots)
                
                # Add filters
                col1, col2, col3 = st.columns(3)
                with col1:
                    team_filter = st.multiselect("Filter by Team", df['Team'].unique(), default=df['Team'].unique())
                with col2:
                    status_filter = st.multiselect("Filter by Status", df['Status'].unique(), default=df['Status'].unique())
                with col3:
                    role_filter = st.multiselect("Filter by Role", df['Role'].unique(), default=df['Role'].unique())
                
                # Apply filters
                filtered_df = df[
                    (df['Team'].isin(team_filter)) &
                    (df['Status'].isin(status_filter)) &
                    (df['Role'].isin(role_filter))
                ]
                
                st.dataframe(filtered_df, use_container_width=True)
        else:
            st.warning("No bot data available. Please check system initialization.")
    
    with tab2:
        st.header("🏢 Organization Structure")
        
        with st.spinner("Loading organization chart..."):
            bot_data = run_async(get_bot_data())
        
        if bot_data:
            # Create hierarchy chart
            hierarchy_chart = create_bot_hierarchy_chart(bot_data)
            st.plotly_chart(hierarchy_chart, use_container_width=True)
            
            # Team breakdown
            st.subheader("📋 Team Breakdown")
            
            for team_name, team_info in bot_data.items():
                with st.expander(f"🔧 {team_name} Team"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Bots", team_info['total_bots'])
                    with col2:
                        st.metric("Active Bots", team_info['active_bots'])
                    with col3:
                        st.metric("Avg Availability", f"{team_info['avg_availability']*100:.1f}%")
                    
                    # Bot details for this team
                    team_bots = team_info['bots']
                    if team_bots:
                        for bot in team_bots:
                            st.write(f"**{bot['name']}** ({bot['role']}) - {bot['status']}")
                            st.write(f"└ Availability: {bot['availability']*100:.1f}% | Tasks: {len(bot['current_tasks'])} | Success: {bot['success_rate']*100:.1f}%")
    
    with tab3:
        st.header("🎯 CEO Decision Center")
        
        if st.session_state.ceo_portal:
            with st.spinner("Loading pending decisions..."):
                try:
                    dashboard_data = run_async(st.session_state.ceo_portal.get_dashboard_data())
                    pending_decisions = dashboard_data.get('pending_decisions', {})
                    
                    # Decision summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Critical", len(pending_decisions.get('critical', [])))
                    with col2:
                        st.metric("High Priority", len(pending_decisions.get('high', [])))
                    with col3:
                        st.metric("Medium Priority", len(pending_decisions.get('medium', [])))
                    with col4:
                        st.metric("Total Pending", sum(len(decisions) for decisions in pending_decisions.values()))
                    
                    # Display decisions by priority
                    for priority, decisions in pending_decisions.items():
                        if decisions:
                            st.subheader(f"🚨 {priority.title()} Priority Decisions")
                            
                            for decision in decisions:
                                with st.expander(f"📋 {decision.get('title', 'Untitled Decision')}"):
                                    st.write(f"**Requesting Bot:** {decision.get('requesting_bot', 'Unknown')}")
                                    st.write(f"**Category:** {decision.get('category', 'Unknown')}")
                                    st.write(f"**Description:** {decision.get('description', 'No description')}")
                                    st.write(f"**Financial Impact:** ${decision.get('financial_impact', 0):,}")
                                    st.write(f"**Risk Level:** {decision.get('risk_level', 0)*100:.1f}%")
                                    st.write(f"**Strategic Alignment:** {decision.get('strategic_alignment', 0)*100:.1f}%")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button(f"✅ Approve", key=f"approve_{decision.get('id')}"):
                                            st.success("Decision approved!")
                                    with col2:
                                        if st.button(f"❌ Reject", key=f"reject_{decision.get('id')}"):
                                            st.error("Decision rejected!")
                
                except Exception as e:
                    st.error(f"Error loading decisions: {str(e)}")
        else:
            st.warning("CEO Portal not initialized")
    
    with tab4:
        st.header("🤝 Cross-Team Coordination")
        
        if st.session_state.coordinator:
            with st.spinner("Loading coordination data..."):
                try:
                    coordination_data = run_async(st.session_state.coordinator.monitor_cross_team_performance())
                    
                    # Coordination metrics
                    metrics = coordination_data.get('cross_team_metrics', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Overall Alignment", f"{metrics.get('overall_alignment', 0)*100:.1f}%")
                    with col2:
                        st.metric("Knowledge Transfer", f"{metrics.get('knowledge_transfer_rate', 0)*100:.1f}%")
                    with col3:
                        st.metric("Synergy Realization", f"{metrics.get('synergy_realization', 0)*100:.1f}%")
                    with col4:
                        st.metric("Coordination Efficiency", f"{metrics.get('coordination_efficiency', 0)*100:.1f}%")
                    
                    # Synergy opportunities
                    st.subheader("💡 Synergy Opportunities")
                    opportunities = run_async(st.session_state.coordinator.identify_synergy_opportunities())
                    
                    if opportunities:
                        for opportunity in opportunities[:5]:  # Show top 5
                            with st.expander(f"🎯 {opportunity.title}"):
                                st.write(f"**Teams Involved:** {', '.join(opportunity.teams_involved)}")
                                st.write(f"**Type:** {opportunity.opportunity_type}")
                                st.write(f"**Description:** {opportunity.description}")
                                st.write(f"**Potential Value:** {opportunity.potential_value*100:.1f}%")
                                st.write(f"**Success Probability:** {opportunity.success_probability*100:.1f}%")
                                st.write(f"**Implementation Effort:** {opportunity.implementation_effort*100:.1f}%")
                    else:
                        st.info("No synergy opportunities identified yet.")
                
                except Exception as e:
                    st.error(f"Error loading coordination data: {str(e)}")
        else:
            st.warning("Coordinator not initialized")
    
    with tab5:
        st.header("🧠 Bot Builder Self-Improvement Center")
        
        st.markdown("""
        **🎯 Recursive AI Enhancement**
        
        This is where Bot Builder achieves true self-awareness by using its own SDLC team to improve itself based on your feedback and requirements.
        """)
        
        # Show current improvement tasks
        if st.session_state.sdlc_team:
            st.subheader("🚀 Active Self-Improvement Tasks")
            
            try:
                # Get tasks related to self-improvement
                improvement_tasks = []
                for task_id, task in st.session_state.sdlc_team.tasks.items():
                    if task.context.get('improvement_type') == 'recursive_enhancement':
                        improvement_tasks.append(task)
                
                if improvement_tasks:
                    for task in improvement_tasks:
                        with st.expander(f"🔧 {task.title}"):
                            st.write(f"**Status:** {task.status.value}")
                            st.write(f"**Progress:** {task.progress_percentage}%")
                            st.write(f"**Assigned Bot:** {task.assigned_bot}")
                            st.write(f"**Priority:** {task.priority.value}")
                            st.write(f"**Description:**")
                            st.write(task.description)
                            
                            # Progress bar
                            st.progress(task.progress_percentage / 100)
                else:
                    st.info("No active self-improvement tasks. Submit a request from the sidebar!")
            
            except Exception as e:
                st.error(f"Error loading improvement tasks: {str(e)}")
        
        # Self-improvement ideas
        st.subheader("💡 Suggested Improvements")
        
        improvement_ideas = [
            "Add real-time notifications for critical decisions",
            "Implement advanced bot performance analytics",
            "Create automated report generation for CEO briefings",
            "Enhance cross-team communication protocols",
            "Add predictive task assignment algorithms",
            "Implement voice command interface for CEO portal",
            "Create mobile app for on-the-go management",
            "Add AI-powered decision recommendation engine"
        ]
        
        for idea in improvement_ideas:
            if st.button(f"🚀 Implement: {idea}", key=f"idea_{idea}"):
                success, message = run_async(
                    submit_self_improvement_request(idea, "MEDIUM")
                )
                if success:
                    st.success(f"✅ Improvement task created for: {idea}")
                else:
                    st.error(f"❌ {message}")
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated:** {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')} | **Status:** {'🟢 Active' if st.session_state.organization_initialized else '🔴 Inactive'}")

if __name__ == "__main__":
    main() 