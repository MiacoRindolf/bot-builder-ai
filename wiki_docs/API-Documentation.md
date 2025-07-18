# API Documentation

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
curl -H "Authorization: Bearer YOUR_API_KEY" \
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
