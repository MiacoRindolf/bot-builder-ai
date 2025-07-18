# core/roles.py
"""
Defines available AI Employee roles and their capabilities.
"""

ROLES = [
    {
        "key": "research_analyst",
        "name": "Research Analyst",
        "description": "Analyzes market data and provides insights.",
        "capabilities": [
            "Market research",
            "Forecasting",
            "Economic analysis"
        ]
    },
    {
        "key": "trader",
        "name": "Trader",
        "description": "Executes trading strategies and manages positions.",
        "capabilities": [
            "Trade execution",
            "Strategy optimization",
            "Order management"
        ]
    },
    {
        "key": "risk_manager",
        "name": "Risk Manager",
        "description": "Monitors and manages risk exposure.",
        "capabilities": [
            "Risk assessment",
            "Scenario analysis",
            "Stress testing"
        ]
    },
    {
        "key": "compliance_officer",
        "name": "Compliance Officer",
        "description": "Ensures regulatory compliance.",
        "capabilities": [
            "Regulatory monitoring",
            "Audit trail management",
            "Policy enforcement"
        ]
    },
    {
        "key": "data_specialist",
        "name": "Data Specialist",
        "description": "Processes and analyzes large datasets.",
        "capabilities": [
            "Data cleaning",
            "Data integration",
            "Data visualization"
        ]
    }
]

def get_role_by_key(key):
    for role in ROLES:
        if role["key"] == key:
            return role
    return None 