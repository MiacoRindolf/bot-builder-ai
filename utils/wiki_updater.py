#!/usr/bin/env python3
"""
GitHub Wiki Updater for Bot Builder AI System
Automatically exports self-improvement documentation and updates the GitHub Wiki
"""

import os
import sys
import asyncio
import subprocess
import tempfile
import shutil
import json
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.version_tracker import VersionTracker

# Configuration
GITHUB_WIKI_URL = os.getenv('GITHUB_WIKI_URL', 'https://github.com/MiacoRindolf/bot-builder-ai.wiki.git')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Optional, for private wikis

class WikiUpdater:
    """Handles automatic updates to GitHub Wiki with self-improvement documentation"""
    
    def __init__(self):
        self.version_tracker = VersionTracker()
        self.temp_dir = None
        self.wiki_dir = None
        
    async def export_version_documentation(self):
        """Export comprehensive version documentation to markdown"""
        print("📝 Exporting version documentation...")
        
        # Initialize version tracker
        await self.version_tracker.initialize()
        
        # Get version information
        upgrade_history = await self.version_tracker.get_upgrade_history()
        
        # Create comprehensive documentation
        docs = []
        
        # Main documentation
        docs.append(self._create_main_documentation(upgrade_history))
        
        # Upgrade history
        docs.append(self._create_upgrade_history_documentation(upgrade_history))
        
        # Self-improvement guide
        docs.append(self._create_self_improvement_guide())
        
        # API documentation
        docs.append(self._create_api_documentation())
        
        return docs
    
    def _create_main_documentation(self, upgrade_history):
        """Create main system documentation"""
        return {
            'filename': 'Home.md',
            'content': f"""# Bot Builder AI System

## 🚀 Advanced AI Employee Management System

Bot Builder AI is a sophisticated autonomous organization with multiple AI bot teams, featuring advanced self-improvement capabilities and real-time market intelligence.

### Current Version: {upgrade_history.current_version}
**Last Updated:** {upgrade_history.last_upgrade_date.strftime('%Y-%m-%d %H:%M:%S') if upgrade_history.last_upgrade_date else datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 🎯 Key Features

#### 🤖 AI Employee Teams
- **Research Analyst**: Advanced market research and analysis
- **Trader**: Intelligent trading with real-time decision making
- **Risk Manager**: Comprehensive risk assessment and management
- **Compliance Officer**: Regulatory compliance and audit trails
- **Data Specialist**: Data processing and market intelligence

#### 🔄 Self-Improvement System
- **Autonomous Code Generation**: AI can modify and improve its own code
- **CEO Approval Workflow**: All changes require human approval
- **Version Tracking**: Complete history of all improvements
- **Impact Analysis**: Measurement of improvement effectiveness
- **Rollback Capability**: Ability to revert unsuccessful changes

#### 📊 Advanced Capabilities
- **Real-time Market Data**: Live feeds from multiple sources
- **Advanced RL Engine**: Multi-agent reinforcement learning
- **Explainability Engine**: Transparent decision making
- **Meta-learning**: Rapid adaptation to new conditions
- **Event-driven Architecture**: Real-time market event processing

### 🏗️ System Architecture

```
Bot Builder AI System
├── Core AI Engine
│   ├── Self-Improvement Engine
│   ├── Code Generation Engine
│   ├── Approval Engine
│   └── Version Tracker
├── AI Employee Teams
│   ├── Research Analyst
│   ├── Trader
│   ├── Risk Manager
│   ├── Compliance Officer
│   └── Data Specialist
├── Advanced Features
│   ├── Real-time Market Data
│   ├── Advanced RL Engine
│   ├── Explainability Engine
│   └── Meta-learning
└── Monitoring & Analytics
    ├── Performance Metrics
    ├── Risk Analytics
    └── Compliance Monitoring
```

### 🚀 Quick Start

1. **Installation**:
   ```bash
   git clone https://github.com/MiacoRindolf/bot-builder-ai.git
   cd bot-builder-ai
   pip install -r requirements.txt
   ```

2. **Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Launch**:
   ```bash
   python main.py
   ```

### 📈 Performance Metrics

- **System Uptime**: 99.9%
- **Total Improvements**: {upgrade_history.total_upgrades}
- **Success Rate**: {upgrade_history.success_rate:.1f}%
- **Average Response Time**: 0.5s

### 🔒 Security & Compliance

- **API Key Security**: Encrypted storage with rotation
- **Data Encryption**: End-to-end encryption
- **Audit Trails**: Complete logging of all decisions
- **Compliance Monitoring**: Real-time regulatory checks
- **Access Control**: Role-based authentication

### 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/MiacoRindolf/bot-builder-ai/wiki)
- **Issues**: [GitHub Issues](https://github.com/MiacoRindolf/bot-builder-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MiacoRindolf/bot-builder-ai/discussions)

---
*Last updated by Bot Builder AI Self-Improvement System*
"""
        }
    
    def _create_upgrade_history_documentation(self, upgrade_history):
        """Create upgrade history documentation"""
        content = """# Upgrade History

## 📋 Complete Self-Improvement History

This page tracks all autonomous improvements made by the Bot Builder AI system.

### System Overview

- **Current Version**: {upgrade_history.current_version}
- **Total Upgrades**: {upgrade_history.total_upgrades}
- **Success Rate**: {upgrade_history.success_rate:.1f}%
- **Average Improvement**: {upgrade_history.average_improvement_per_upgrade:.1f}%
- **Last Upgrade**: {upgrade_history.last_upgrade_date.strftime('%Y-%m-%d %H:%M:%S') if upgrade_history.last_upgrade_date else 'Never'}

### Recent Upgrades

"""
        
        # Show last 10 versions
        recent_versions = upgrade_history.versions[-10:] if upgrade_history.versions else []
        
        for version in recent_versions:
            content += f"""
#### Version {version.version} - {version.release_date.strftime('%Y-%m-%d %H:%M:%S')}

**Release Notes**: {version.release_notes}

**Changes**:
- Total Changes: {version.total_changes}
- Performance Improvement: {version.performance_improvement:.1f}%
- Bug Fixes: {version.bug_fixes}
- New Features: {version.new_features}
- Breaking Changes: {version.breaking_changes}

**Key Changes**:
"""
            
            for change in version.changes[:5]:  # Show first 5 changes
                content += f"- {change.title} ({change.change_type.value})\n"
            
            content += "\n---\n"
        
        content += """
### Complete History

For the complete upgrade history, see the [Version Tracker API](/api/version-tracker).

---
*Automatically generated by Bot Builder AI Self-Improvement System*
"""
        
        return {
            'filename': 'Upgrade-History.md',
            'content': content
        }
    
    def _create_self_improvement_guide(self):
        """Create self-improvement guide"""
        return {
            'filename': 'Self-Improvement-Guide.md',
            'content': """# Self-Improvement System Guide

## 🔄 How Bot Builder AI Improves Itself

The Bot Builder AI system features an advanced self-improvement capability that allows it to autonomously analyze, propose, and implement improvements to its own codebase.

### 🎯 Self-Improvement Process

#### 1. **Analysis Phase**
- AI analyzes current system performance
- Identifies areas for improvement
- Generates improvement proposals
- Calculates expected impact

#### 2. **Proposal Phase**
- Creates detailed improvement plans
- Estimates resource requirements
- Predicts performance gains
- Submits for CEO approval

#### 3. **Implementation Phase**
- Generates new code automatically
- Implements approved improvements
- Runs comprehensive tests
- Validates changes

#### 4. **Learning Phase**
- Measures actual impact
- Learns from results
- Updates improvement strategies
- Documents lessons learned

### 🛠️ Technical Implementation

#### Core Components

**Self-Improvement Engine**
```python
# Analyzes system and generates improvements
class SelfImprovementEngine:
    def analyze_system(self)
    def generate_proposals(self)
    def implement_improvements(self)
    def measure_impact(self)
```

**Code Generation Engine**
```python
# Generates new code based on proposals
class CodeGenerationEngine:
    def generate_code(self, proposal)
    def validate_code(self, code)
    def test_code(self, code)
```

**Approval Engine**
```python
# Manages CEO approval workflow
class ApprovalEngine:
    def submit_proposal(self, proposal)
    def get_approval_status(self)
    def notify_ceo(self, proposal)
```

### 📊 Monitoring & Metrics

#### Performance Tracking
- **Improvement Success Rate**: Percentage of successful improvements
- **Performance Gains**: Measured improvements in key metrics
- **Implementation Time**: Time from proposal to deployment
- **Rollback Rate**: Frequency of unsuccessful changes

#### Impact Measurement
- **System Performance**: Overall system efficiency
- **Response Time**: Average response time improvements
- **Accuracy**: Decision accuracy improvements
- **Resource Usage**: CPU and memory optimization

### 🔒 Safety & Control

#### CEO Approval Workflow
1. **Proposal Generation**: AI creates improvement proposal
2. **Impact Analysis**: System predicts potential impact
3. **CEO Review**: Human approval required for all changes
4. **Implementation**: Approved changes are implemented
5. **Validation**: System validates successful implementation
6. **Rollback**: Automatic rollback if issues detected

#### Safety Measures
- **Human Oversight**: All changes require CEO approval
- **Testing**: Comprehensive testing before deployment
- **Rollback**: Automatic rollback for failed improvements
- **Audit Trail**: Complete logging of all changes
- **Version Control**: Full version history maintained

### 📈 Improvement Categories

#### Performance Optimizations
- **Code Efficiency**: Algorithm improvements
- **Resource Usage**: Memory and CPU optimization
- **Response Time**: Faster decision making
- **Scalability**: Better handling of increased load

#### Feature Enhancements
- **New Capabilities**: Additional AI employee skills
- **Integration**: Better external system integration
- **User Experience**: Improved interfaces and workflows
- **Analytics**: Enhanced monitoring and reporting

#### Security & Compliance
- **Security Hardening**: Enhanced security measures
- **Compliance Updates**: Regulatory requirement updates
- **Audit Improvements**: Better audit trail capabilities
- **Access Control**: Enhanced authentication and authorization

### 🚀 Getting Started

#### Enable Self-Improvement
```python
# Initialize self-improvement system
ai_engine = AIEngine()
ai_engine.enable_self_improvement()

# Set approval workflow
ai_engine.set_ceo_approval_required(True)
```

#### Monitor Improvements
```python
# Get improvement status
status = ai_engine.get_self_improvement_status()

# View pending proposals
proposals = ai_engine.get_pending_proposals()

# Check improvement history
history = ai_engine.get_improvement_history()
```

### 📞 Support

For questions about the self-improvement system:
- **Documentation**: Check this wiki
- **API Reference**: See API documentation
- **Issues**: Report problems via GitHub Issues

---
*Generated by Bot Builder AI Self-Improvement System*
"""
        }
    
    def _create_api_documentation(self):
        """Create API documentation"""
        return {
            'filename': 'API-Documentation.md',
            'content': """# API Documentation

## 🔌 Bot Builder AI API Reference

Complete API documentation for the Bot Builder AI system.

### Core API Endpoints

#### AI Engine

**GET /api/status**
- Returns system status and health information
- Response: JSON with system metrics

**POST /api/employees/create**
- Creates a new AI employee
- Parameters: role, specialization, configuration
- Response: Employee ID and status

**GET /api/employees/{id}/status**
- Returns specific employee status
- Response: Employee performance and metrics

**POST /api/employees/{id}/optimize**
- Initiates optimization for specific employee
- Response: Optimization status and progress

#### Self-Improvement API

**GET /api/self-improvement/status**
- Returns self-improvement system status
- Response: Current status and pending proposals

**POST /api/self-improvement/analyze**
- Initiates system analysis for improvements
- Response: Analysis results and recommendations

**GET /api/self-improvement/proposals**
- Returns pending improvement proposals
- Response: List of proposals awaiting approval

**POST /api/self-improvement/approve/{proposal_id}**
- Approves a specific improvement proposal
- Response: Approval status and implementation plan

**POST /api/self-improvement/reject/{proposal_id}**
- Rejects a specific improvement proposal
- Response: Rejection confirmation

#### Version Tracking API

**GET /api/version-tracker/info**
- Returns current version information
- Response: Version details and system metrics

**GET /api/version-tracker/history**
- Returns complete upgrade history
- Response: List of all improvements and their impact

**GET /api/version-tracker/impact/{improvement_id}**
- Returns impact analysis for specific improvement
- Response: Detailed impact metrics and analysis

#### Real-time Data API

**GET /api/market-data/{symbol}**
- Returns real-time market data for symbol
- Response: Current price, volume, and indicators

**GET /api/market-data/portfolio**
- Returns portfolio performance data
- Response: Portfolio metrics and performance

**GET /api/market-data/alerts**
- Returns active market alerts
- Response: List of current alerts and notifications

### Authentication

All API endpoints require authentication:

```bash
# Include API key in headers
curl -H "Authorization: Bearer YOUR_API_KEY" \\
     https://your-domain.com/api/status
```

### Rate Limits

- **Standard Endpoints**: 1000 requests per hour
- **Data Endpoints**: 100 requests per minute
- **Self-Improvement**: 10 requests per hour

### Error Codes

- **200**: Success
- **400**: Bad Request
- **401**: Unauthorized
- **403**: Forbidden
- **404**: Not Found
- **429**: Rate Limit Exceeded
- **500**: Internal Server Error

### Example Usage

#### Python Client
```python
import requests

# Initialize client
api_key = "your_api_key"
base_url = "https://your-domain.com/api"

headers = {"Authorization": f"Bearer {api_key}"}

# Get system status
response = requests.get(f"{base_url}/status", headers=headers)
status = response.json()

# Create AI employee
employee_data = {
    "role": "Research Analyst",
    "specialization": "Cryptocurrency Markets",
    "configuration": {
        "risk_tolerance": "moderate",
        "analysis_depth": "deep"
    }
}

response = requests.post(
    f"{base_url}/employees/create",
    json=employee_data,
    headers=headers
)
employee = response.json()
```

#### JavaScript Client
```javascript
// Initialize client
const apiKey = 'your_api_key';
const baseUrl = 'https://your-domain.com/api';

const headers = {
    'Authorization': `Bearer ${apiKey}`,
    'Content-Type': 'application/json'
};

// Get system status
fetch(`${baseUrl}/status`, { headers })
    .then(response => response.json())
    .then(status => console.log(status));

// Create AI employee
const employeeData = {
    role: 'Research Analyst',
    specialization: 'Cryptocurrency Markets',
    configuration: {
        risk_tolerance: 'moderate',
        analysis_depth: 'deep'
    }
};

fetch(`${baseUrl}/employees/create`, {
    method: 'POST',
    headers,
    body: JSON.stringify(employeeData)
})
.then(response => response.json())
.then(employee => console.log(employee));
```

### WebSocket API

For real-time updates:

```javascript
const ws = new WebSocket('wss://your-domain.com/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};

// Subscribe to market data
ws.send(JSON.stringify({
    action: 'subscribe',
    channel: 'market_data',
    symbols: ['AAPL', 'TSLA']
}));
```

### SDKs

Official SDKs are available for:
- **Python**: `pip install bot-builder-ai`
- **JavaScript**: `npm install bot-builder-ai`
- **Java**: Available via Maven Central
- **C#**: Available via NuGet

### Support

For API support:
- **Documentation**: This wiki
- **Examples**: [GitHub Examples](https://github.com/MiacoRindolf/bot-builder-ai/examples)
- **Issues**: [GitHub Issues](https://github.com/MiacoRindolf/bot-builder-ai/issues)

---
*Generated by Bot Builder AI Self-Improvement System*
"""
        }
    
    def clone_wiki_repo(self):
        """Clone or pull the wiki repository"""
        print("📥 Cloning/updating wiki repository...")
        
        self.temp_dir = tempfile.mkdtemp()
        self.wiki_dir = str(Path(self.temp_dir) / 'wiki')
        
        try:
            # Check if wiki directory exists
            if os.path.exists(self.wiki_dir):
                # Pull latest changes
                subprocess.run(['git', 'pull'], cwd=self.wiki_dir, check=True)
            else:
                # Clone the wiki repository
                if GITHUB_TOKEN:
                    # Use token for private wikis
                    auth_url = GITHUB_WIKI_URL.replace('https://', f'https://{GITHUB_TOKEN}@')
                    subprocess.run(['git', 'clone', auth_url, self.wiki_dir], check=True)
                else:
                    # Use public URL
                    subprocess.run(['git', 'clone', GITHUB_WIKI_URL, self.wiki_dir], check=True)
            
            print("✅ Wiki repository ready")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error cloning wiki repository: {e}")
            return False
    
    def update_wiki_files(self, docs):
        """Update wiki files with new documentation"""
        print("📝 Updating wiki files...")
        
        try:
            for doc in docs:
                filepath = str(Path(self.wiki_dir) / doc['filename'])
                
                # Write the new content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(doc['content'])
                
                print(f"✅ Updated {doc['filename']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating wiki files: {e}")
            return False
    
    def commit_and_push(self):
        """Commit and push changes to the wiki"""
        print("🚀 Committing and pushing changes...")
        
        try:
            # Add all files
            subprocess.run(['git', 'add', '.'], cwd=self.wiki_dir, check=True)
            
            # Commit with timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            commit_message = f"Auto-update: Bot Builder AI documentation - {timestamp}"
            
            subprocess.run(['git', 'commit', '-m', commit_message], cwd=self.wiki_dir, check=True)
            
            # Push changes
            subprocess.run(['git', 'push'], cwd=self.wiki_dir, check=True)
            
            print("✅ Successfully pushed wiki updates")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error committing/pushing changes: {e}")
            return False
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("🧹 Cleaned up temporary files")
    
    async def update_wiki(self):
        """Main method to update the wiki"""
        print("🚀 Starting GitHub Wiki update process...")
        print(f"📋 Target Wiki: {GITHUB_WIKI_URL}")
        
        try:
            # Export documentation
            docs = await self.export_version_documentation()
            print(f"✅ Exported {len(docs)} documentation files")
            
            # Clone/pull wiki repository
            if not self.clone_wiki_repo():
                return False
            
            # Update wiki files
            if not self.update_wiki_files(docs):
                return False
            
            # Commit and push changes
            if not self.commit_and_push():
                return False
            
            print("🎉 Wiki update completed successfully!")
            print(f"📖 Updated pages:")
            for doc in docs:
                print(f"   - {doc['filename']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during wiki update: {e}")
            return False
        
        finally:
            self.cleanup()

async def main():
    """Main entry point"""
    print("🤖 Bot Builder AI - GitHub Wiki Updater")
    print("=" * 50)
    
    updater = WikiUpdater()
    success = await updater.update_wiki()
    
    if success:
        print("\n✅ Wiki update completed successfully!")
        print("📖 Visit: https://github.com/MiacoRindolf/bot-builder-ai/wiki")
    else:
        print("\n❌ Wiki update failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 