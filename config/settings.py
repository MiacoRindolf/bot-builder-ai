"""
Configuration settings for the Bot Builder AI system.
Handles environment variables, default values, and configuration validation.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, validator
import logging

# Load environment variables from .env file
# Try multiple possible locations for the .env file
env_files = ['.env', 'environment_config.txt']
env_loaded = False

for env_file in env_files:
    if os.path.exists(env_file):
        load_dotenv(env_file)
        env_loaded = True
        break

if not env_loaded:
    print("⚠️  Warning: No .env file found. Using default configuration.")
    print("   Run 'python setup_env.py' to create the environment configuration.")

class Settings(BaseModel):
    """Main settings class for the Bot Builder AI system."""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///bot_builder.db")
    
    # Redis Configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_s3_bucket: str = os.getenv("AWS_S3_BUCKET", "bot-builder-data")
    
    # Security Configuration
    secret_key: str = os.getenv("SECRET_KEY", "change_this_in_production")
    encryption_key: str = os.getenv("ENCRYPTION_KEY", "change_this_in_production")
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "change_this_in_production")
    
    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "logs/bot_builder.log")
    
    # AI Employee Configuration
    max_ai_employees: int = int(os.getenv("MAX_AI_EMPLOYEES", "50"))
    training_timeout_hours: int = int(os.getenv("TRAINING_TIMEOUT_HOURS", "24"))
    optimization_interval_hours: int = int(os.getenv("OPTIMIZATION_INTERVAL_HOURS", "6"))
    
    # Financial Data Sources
    alpha_vantage_api_key: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    quandl_api_key: Optional[str] = os.getenv("QUANDL_API_KEY")
    yahoo_finance_enabled: bool = os.getenv("YAHOO_FINANCE_ENABLED", "true").lower() == "true"
    
    # Monitoring Configuration
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # Compliance Configuration
    compliance_mode: str = os.getenv("COMPLIANCE_MODE", "strict")
    audit_trail_enabled: bool = os.getenv("AUDIT_TRAIL_ENABLED", "true").lower() == "true"
    regulatory_framework: str = os.getenv("REGULATORY_FRAMEWORK", "SEC")
    
    # Performance Configuration
    max_concurrent_trainings: int = int(os.getenv("MAX_CONCURRENT_TRAININGS", "5"))
    max_concurrent_trades: int = int(os.getenv("MAX_CONCURRENT_TRADES", "10"))
    risk_limit_percentage: float = float(os.getenv("RISK_LIMIT_PERCENTAGE", "2.0"))
    
    # UI Configuration
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    gradio_port: int = int(os.getenv("GRADIO_PORT", "7860"))
    enable_websockets: bool = os.getenv("ENABLE_WEBSOCKETS", "true").lower() == "true"
    
    # Development Configuration
    debug_mode: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    enable_hot_reload: bool = os.getenv("ENABLE_HOT_RELOAD", "true").lower() == "true"
    test_mode: bool = os.getenv("TEST_MODE", "false").lower() == "true"
    
    # AI Employee Role Configurations
    ai_employee_roles: Dict[str, Dict[str, Any]] = {
        "research_analyst": {
            "model_architecture": "transformer",
            "training_data_sources": ["financial_news", "market_data", "social_sentiment"],
            "evaluation_metrics": ["accuracy", "sharpe_ratio", "information_ratio"],
            "optimization_focus": ["prediction_accuracy", "risk_adjustment"],
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        },
        "trader": {
            "model_architecture": "reinforcement_learning",
            "training_data_sources": ["market_data", "order_book", "execution_data"],
            "evaluation_metrics": ["execution_speed", "win_rate", "profit_loss"],
            "optimization_focus": ["execution_speed", "risk_management"],
            "learning_rate": 0.0001,
            "batch_size": 64,
            "epochs": 200
        },
        "risk_manager": {
            "model_architecture": "statistical",
            "training_data_sources": ["market_data", "portfolio_data", "risk_metrics"],
            "evaluation_metrics": ["var_accuracy", "stress_test_pass_rate", "risk_adjustment"],
            "optimization_focus": ["risk_prediction", "scenario_analysis"],
            "learning_rate": 0.01,
            "batch_size": 128,
            "epochs": 50
        },
        "compliance_officer": {
            "model_architecture": "nlp_transformer",
            "training_data_sources": ["regulatory_documents", "transaction_data", "audit_logs"],
            "evaluation_metrics": ["compliance_score", "audit_accuracy", "explainability"],
            "optimization_focus": ["regulatory_compliance", "explainability"],
            "learning_rate": 0.0005,
            "batch_size": 16,
            "epochs": 150
        },
        "data_specialist": {
            "model_architecture": "autoencoder",
            "training_data_sources": ["raw_market_data", "cleaned_data", "metadata"],
            "evaluation_metrics": ["data_quality_score", "cleaning_accuracy", "processing_speed"],
            "optimization_focus": ["data_quality", "processing_efficiency"],
            "learning_rate": 0.002,
            "batch_size": 256,
            "epochs": 75
        }
    }
    
    @validator("openai_api_key")
    def validate_openai_api_key(cls, v):
        """Validate that OpenAI API key is provided."""
        if not v:
            raise ValueError("OpenAI API key is required")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator("compliance_mode")
    def validate_compliance_mode(cls, v):
        """Validate compliance mode."""
        valid_modes = ["strict", "moderate", "relaxed"]
        if v.lower() not in valid_modes:
            raise ValueError(f"Compliance mode must be one of {valid_modes}")
        return v.lower()
    
    @validator("regulatory_framework")
    def validate_regulatory_framework(cls, v):
        """Validate regulatory framework."""
        valid_frameworks = ["SEC", "FINRA", "CFTC", "EU_MIFID", "UK_FCA"]
        if v.upper() not in valid_frameworks:
            raise ValueError(f"Regulatory framework must be one of {valid_frameworks}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings

def setup_logging():
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.log_file),
            logging.StreamHandler()
        ]
    )

def validate_configuration():
    """Validate the configuration and return any issues."""
    issues = []
    
    # Check required API keys
    if not settings.openai_api_key:
        issues.append("OpenAI API key is required")
    
    # Check database connection
    if not settings.database_url:
        issues.append("Database URL is required")
    
    # Check security keys
    if settings.secret_key == "change_this_in_production":
        issues.append("Secret key should be changed in production")
    
    if settings.encryption_key == "change_this_in_production":
        issues.append("Encryption key should be changed in production")
    
    # Check performance limits
    if settings.max_ai_employees <= 0:
        issues.append("Maximum AI employees must be greater than 0")
    
    if settings.risk_limit_percentage <= 0 or settings.risk_limit_percentage > 100:
        issues.append("Risk limit percentage must be between 0 and 100")
    
    return issues

# Setup logging when module is imported
setup_logging()

# Validate configuration
config_issues = validate_configuration()
if config_issues:
    logging.warning("Configuration issues found:")
    for issue in config_issues:
        logging.warning(f"  - {issue}") 