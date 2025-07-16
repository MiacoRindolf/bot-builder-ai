# Bot Builder AI - Setup Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- At least 4GB RAM (8GB recommended)
- Internet connection for API calls

### Step 1: Clone and Setup
```bash
# Clone the repository (if using git)
git clone <repository-url>
cd BotBuilder

# Or if you have the files directly, navigate to the BotBuilder directory
cd BotBuilder
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# Copy the example environment file
cp env_example.txt .env

# Edit the .env file with your settings
# At minimum, you need to set your OpenAI API key
```

### Step 5: Set Your OpenAI API Key
Edit the `.env` file and set your OpenAI API key:
```
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### Step 6: Run the System
```bash
# Use the startup script (recommended)
python start.py

# Or run directly:
# For Streamlit interface:
streamlit run ui/streamlit_app.py

# For Gradio interface:
python ui/gradio_app.py

# For API/backend only:
python main.py --mode api
```

## 📋 Detailed Setup Instructions

### Environment Configuration

The `.env` file contains all configuration settings. Here are the key ones:

#### Required Settings
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `DATABASE_URL`: Database connection (defaults to SQLite)
- `SECRET_KEY`: Secret key for security (change in production)

#### Optional Settings
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MAX_AI_EMPLOYEES`: Maximum number of AI Employees (default: 50)
- `STREAMLIT_PORT`: Port for Streamlit interface (default: 8501)
- `GRADIO_PORT`: Port for Gradio interface (default: 7860)

### Database Setup

The system uses SQLite by default, which requires no additional setup. For production:

#### PostgreSQL
```bash
# Install PostgreSQL dependencies
pip install psycopg2-binary

# Set DATABASE_URL in .env
DATABASE_URL=postgresql://user:password@localhost/bot_builder
```

#### MySQL
```bash
# Install MySQL dependencies
pip install mysqlclient

# Set DATABASE_URL in .env
DATABASE_URL=mysql://user:password@localhost/bot_builder
```

### Redis Setup (Optional)

For better performance and caching:

```bash
# Install Redis
# On Ubuntu/Debian:
sudo apt-get install redis-server

# On macOS:
brew install redis

# Start Redis
redis-server

# Set REDIS_URL in .env
REDIS_URL=redis://localhost:6379
```

## 🔧 System Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux

### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 10GB+ free space
- **GPU**: NVIDIA GPU with CUDA support (for faster AI training)

### Performance Optimization

For better performance:

1. **Use SSD storage** for faster data access
2. **Increase RAM** for larger datasets
3. **Use GPU acceleration** for AI model training
4. **Enable Redis caching** for faster responses
5. **Use production database** (PostgreSQL/MySQL) for large deployments

## 🐛 Troubleshooting

### Common Issues

#### 1. OpenAI API Key Error
```
Error: OpenAI API key is required
```
**Solution**: Set your OpenAI API key in the `.env` file

#### 2. Import Errors
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

#### 3. Port Already in Use
```
Error: Port 8501 is already in use
```
**Solution**: Change the port in `.env` file or kill the process using the port

#### 4. Memory Issues
```
MemoryError: Unable to allocate memory
```
**Solution**: 
- Close other applications
- Reduce `MAX_AI_EMPLOYEES` in `.env`
- Use a machine with more RAM

#### 5. Database Connection Issues
```
Error: Unable to connect to database
```
**Solution**: 
- Check database URL in `.env`
- Ensure database server is running
- Verify credentials

### Performance Issues

#### Slow Response Times
1. Check internet connection
2. Verify OpenAI API key is valid
3. Enable Redis caching
4. Reduce concurrent operations

#### High Memory Usage
1. Reduce `MAX_AI_EMPLOYEES`
2. Clear cache regularly
3. Restart the system periodically
4. Monitor with system tools

### Log Files

Logs are stored in `logs/bot_builder.log`. Check this file for detailed error information:

```bash
# View recent logs
tail -f logs/bot_builder.log

# Search for errors
grep ERROR logs/bot_builder.log
```

## 🔒 Security Considerations

### Production Deployment

1. **Change default secrets**:
   ```
   SECRET_KEY=your_secure_secret_key
   ENCRYPTION_KEY=your_secure_encryption_key
   JWT_SECRET_KEY=your_secure_jwt_key
   ```

2. **Use HTTPS** in production

3. **Set up proper authentication**

4. **Regular security updates**

5. **Monitor access logs**

### API Key Security

- Never commit API keys to version control
- Use environment variables
- Rotate keys regularly
- Monitor API usage

## 📊 Monitoring and Maintenance

### System Monitoring

The system includes built-in monitoring:

1. **Performance Metrics**: Track AI Employee performance
2. **System Health**: Monitor CPU, memory, disk usage
3. **Error Tracking**: Log and alert on errors
4. **Usage Analytics**: Track system usage patterns

### Regular Maintenance

1. **Update dependencies**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Clean old data**:
   ```bash
   # Clear old logs
   find logs/ -name "*.log" -mtime +30 -delete
   
   # Clear old cache
   rm -rf data/storage/cache/*
   ```

3. **Backup data**:
   ```bash
   # Backup database
   sqlite3 bot_builder.db .dump > backup.sql
   
   # Backup configuration
   cp .env .env.backup
   ```

## 🚀 Deployment Options

### Local Development
```bash
python start.py
```

### Docker Deployment
```dockerfile
# Create Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment

#### AWS
```bash
# Deploy to AWS Lambda
serverless deploy

# Deploy to AWS ECS
aws ecs create-service --cluster bot-builder --service-name bot-builder-service
```

#### Google Cloud
```bash
# Deploy to Google Cloud Run
gcloud run deploy bot-builder --source .
```

#### Azure
```bash
# Deploy to Azure App Service
az webapp up --name bot-builder --runtime python:3.9
```

## 📞 Support

### Getting Help

1. **Check the logs**: `logs/bot_builder.log`
2. **Review documentation**: README.md
3. **Check configuration**: `.env` file
4. **Verify dependencies**: `requirements.txt`

### Common Commands

```bash
# Check system status
python -c "from config.settings import settings; print('Configuration loaded successfully')"

# Test OpenAI connection
python -c "import openai; openai.api_key='your_key'; print('OpenAI connection successful')"

# View system info
python -c "import sys; print(f'Python {sys.version}')"
```

### Performance Testing

```bash
# Test system performance
python -c "
import time
start = time.time()
# Your test code here
print(f'Execution time: {time.time() - start:.2f}s')
"
```

## 🔄 Updates and Upgrades

### Updating the System

1. **Backup your data**
2. **Update code** (if using git)
3. **Update dependencies**: `pip install --upgrade -r requirements.txt`
4. **Test the system**
5. **Restart services**

### Version Compatibility

- Python 3.8-3.11 supported
- OpenAI API v1 compatible
- Streamlit 1.28+ required
- Gradio 3.40+ required

## 📈 Scaling Considerations

### Horizontal Scaling

1. **Load Balancing**: Use multiple instances
2. **Database Sharding**: Distribute data across databases
3. **Caching**: Use Redis cluster
4. **Microservices**: Split into separate services

### Vertical Scaling

1. **Increase RAM**: For larger datasets
2. **Add CPU cores**: For parallel processing
3. **Use GPU**: For AI model training
4. **SSD storage**: For faster I/O

## 🎯 Best Practices

### Development

1. **Use virtual environments**
2. **Follow PEP 8 style guide**
3. **Write tests for new features**
4. **Document your changes**
5. **Use version control**

### Production

1. **Monitor system resources**
2. **Set up automated backups**
3. **Use proper logging**
4. **Implement error handling**
5. **Regular security audits**

### AI Employee Management

1. **Start with simple roles**
2. **Monitor performance closely**
3. **Optimize based on metrics**
4. **Regular retraining**
5. **A/B testing for improvements**

---

For additional support or questions, please refer to the main README.md file or create an issue in the project repository. 