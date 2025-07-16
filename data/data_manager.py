"""
Data Manager for the Bot Builder AI system.
Handles data retrieval, storage, and management for AI Employees.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

class DataManager:
    """
    Data Manager for handling all data operations in the Bot Builder AI system.
    
    Responsibilities:
    - Market data retrieval and caching
    - Training data preparation
    - Data storage and management
    - Data quality assessment
    - Real-time data feeds
    """
    
    def __init__(self):
        """Initialize the Data Manager."""
        self.cache = {}
        self.data_sources = {
            "market_data": self._get_market_data_source(),
            "financial_news": self._get_news_data_source(),
            "social_sentiment": self._get_sentiment_data_source(),
            "regulatory_documents": self._get_regulatory_data_source(),
            "execution_data": self._get_execution_data_source()
        }
        
        # Create data directories
        self.data_dir = Path("data/storage")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps = {}
        
        logger.info("Data Manager initialized")
    
    async def get_market_data(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Market symbol (e.g., 'AAPL', 'BTC-USD')
            days_back: Number of days of historical data
            
        Returns:
            DataFrame with market data
        """
        try:
            cache_key = f"market_data_{symbol}_{days_back}"
            
            # Check cache
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Get data from source
            data = await self._fetch_market_data(symbol, days_back)
            
            # Cache the data
            self._cache_data(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental data for a symbol.
        
        Args:
            symbol: Market symbol
            
        Returns:
            Dictionary with fundamental data
        """
        try:
            cache_key = f"fundamental_data_{symbol}"
            
            # Check cache
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Get data from source
            data = await self._fetch_fundamental_data(symbol)
            
            # Cache the data
            self._cache_data(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting fundamental data for {symbol}: {str(e)}")
            return {}
    
    async def get_training_data(self, data_source: str, specialization: str) -> Dict[str, Any]:
        """
        Get training data for AI Employee training.
        
        Args:
            data_source: Type of data source
            specialization: Specialization area
            
        Returns:
            Dictionary with training data
        """
        try:
            cache_key = f"training_data_{data_source}_{specialization}"
            
            # Check cache
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Get data based on source type
            if data_source == "financial_news":
                data = await self._get_news_training_data(specialization)
            elif data_source == "market_data":
                data = await self._get_market_training_data(specialization)
            elif data_source == "social_sentiment":
                data = await self._get_sentiment_training_data(specialization)
            elif data_source == "regulatory_documents":
                data = await self._get_regulatory_training_data(specialization)
            elif data_source == "execution_data":
                data = await self._get_execution_training_data(specialization)
            else:
                data = await self._get_generic_training_data(data_source, specialization)
            
            # Cache the data
            self._cache_data(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting training data for {data_source}: {str(e)}")
            return {}
    
    async def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time market data.
        
        Args:
            symbol: Market symbol
            
        Returns:
            Dictionary with real-time data
        """
        try:
            # Real-time data is not cached
            data = await self._fetch_real_time_data(symbol)
            return data
            
        except Exception as e:
            logger.error(f"Error getting real-time data for {symbol}: {str(e)}")
            return {}
    
    async def store_employee_data(self, employee_id: str, data: Dict[str, Any]) -> bool:
        """
        Store data for a specific AI Employee.
        
        Args:
            employee_id: AI Employee ID
            data: Data to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.data_dir / f"employee_{employee_id}.json"
            
            # Load existing data
            existing_data = {}
            if file_path.exists():
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            
            # Update with new data
            existing_data.update(data)
            existing_data['last_updated'] = datetime.now().isoformat()
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            logger.info(f"Stored data for employee {employee_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing data for employee {employee_id}: {str(e)}")
            return False
    
    async def get_employee_data(self, employee_id: str) -> Dict[str, Any]:
        """
        Get stored data for a specific AI Employee.
        
        Args:
            employee_id: AI Employee ID
            
        Returns:
            Dictionary with employee data
        """
        try:
            file_path = self.data_dir / f"employee_{employee_id}.json"
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting data for employee {employee_id}: {str(e)}")
            return {}
    
    async def assess_data_quality(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess the quality of data.
        
        Args:
            data: Data to assess
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            if isinstance(data, pd.DataFrame):
                return self._assess_dataframe_quality(data)
            elif isinstance(data, dict):
                return self._assess_dict_quality(data)
            else:
                return {"error": "Unsupported data type"}
                
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            return {"error": str(e)}
    
    async def clean_data(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Clean and preprocess data.
        
        Args:
            data: Data to clean
            
        Returns:
            Cleaned data
        """
        try:
            if isinstance(data, pd.DataFrame):
                return self._clean_dataframe(data)
            elif isinstance(data, dict):
                return self._clean_dict(data)
            else:
                return data
                
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return data
    
    def _get_market_data_source(self) -> str:
        """Get market data source configuration."""
        if settings.yahoo_finance_enabled:
            return "yahoo_finance"
        elif settings.alpha_vantage_api_key:
            return "alpha_vantage"
        else:
            return "mock"
    
    def _get_news_data_source(self) -> str:
        """Get news data source configuration."""
        return "mock"  # Placeholder
    
    def _get_sentiment_data_source(self) -> str:
        """Get sentiment data source configuration."""
        return "mock"  # Placeholder
    
    def _get_regulatory_data_source(self) -> str:
        """Get regulatory data source configuration."""
        return "mock"  # Placeholder
    
    def _get_execution_data_source(self) -> str:
        """Get execution data source configuration."""
        return "mock"  # Placeholder
    
    async def _fetch_market_data(self, symbol: str, days_back: int) -> pd.DataFrame:
        """Fetch market data from source."""
        try:
            # Simulate data fetching
            await asyncio.sleep(0.1)
            
            # Generate mock data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate realistic price data
            base_price = 100.0
            price_data = []
            
            for i, date in enumerate(dates):
                # Add some randomness to price movement
                change = np.random.normal(0, 2)  # 2% daily volatility
                base_price *= (1 + change / 100)
                
                price_data.append({
                    'Date': date,
                    'Open': base_price * (1 + np.random.normal(0, 0.01)),
                    'High': base_price * (1 + abs(np.random.normal(0, 0.02))),
                    'Low': base_price * (1 - abs(np.random.normal(0, 0.02))),
                    'Close': base_price,
                    'Volume': int(np.random.uniform(1000000, 10000000))
                })
            
            df = pd.DataFrame(price_data)
            df.set_index('Date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return pd.DataFrame()
    
    async def _fetch_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data from source."""
        try:
            # Simulate data fetching
            await asyncio.sleep(0.1)
            
            # Generate mock fundamental data
            return {
                'symbol': symbol,
                'pe_ratio': np.random.uniform(10, 30),
                'pb_ratio': np.random.uniform(1, 5),
                'dividend_yield': np.random.uniform(0, 0.05),
                'market_cap': np.random.uniform(1e9, 1e12),
                'revenue': np.random.uniform(1e8, 1e11),
                'profit_margin': np.random.uniform(0.05, 0.25),
                'debt_to_equity': np.random.uniform(0.1, 1.0),
                'current_ratio': np.random.uniform(1.0, 3.0)
            }
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data: {str(e)}")
            return {}
    
    async def _fetch_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time market data."""
        try:
            # Simulate real-time data
            await asyncio.sleep(0.05)
            
            base_price = 100.0
            change = np.random.normal(0, 1)
            current_price = base_price * (1 + change / 100)
            
            return {
                'symbol': symbol,
                'price': current_price,
                'change': change,
                'change_percent': change,
                'volume': int(np.random.uniform(100000, 1000000)),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time data: {str(e)}")
            return {}
    
    async def _get_news_training_data(self, specialization: str) -> Dict[str, Any]:
        """Get news training data."""
        try:
            # Simulate news data
            await asyncio.sleep(0.1)
            
            news_items = [
                f"Market analysis shows positive trends in {specialization}",
                f"New regulations impact {specialization} sector",
                f"Technology breakthrough in {specialization} industry",
                f"Economic indicators suggest growth in {specialization}",
                f"Expert opinions on {specialization} market outlook"
            ]
            
            return {
                'texts': news_items,
                'sentiments': [0.8, -0.2, 0.9, 0.6, 0.7],
                'categories': ['analysis', 'regulation', 'technology', 'economics', 'expert']
            }
            
        except Exception as e:
            logger.error(f"Error getting news training data: {str(e)}")
            return {}
    
    async def _get_market_training_data(self, specialization: str) -> Dict[str, Any]:
        """Get market training data."""
        try:
            # Get market data for training
            market_data = await self.get_market_data(specialization, days_back=365)
            
            return {
                'prices': market_data['Close'].tolist(),
                'volumes': market_data['Volume'].tolist(),
                'dates': market_data.index.tolist(),
                'returns': market_data['Close'].pct_change().dropna().tolist()
            }
            
        except Exception as e:
            logger.error(f"Error getting market training data: {str(e)}")
            return {}
    
    async def _get_sentiment_training_data(self, specialization: str) -> Dict[str, Any]:
        """Get sentiment training data."""
        try:
            # Simulate sentiment data
            await asyncio.sleep(0.1)
            
            return {
                'texts': [
                    f"Great news for {specialization} investors!",
                    f"Concerns about {specialization} market stability",
                    f"Positive outlook for {specialization} sector",
                    f"Mixed reactions to {specialization} developments"
                ],
                'sentiments': [0.9, -0.6, 0.8, 0.1],
                'sources': ['twitter', 'reddit', 'news', 'forum']
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment training data: {str(e)}")
            return {}
    
    async def _get_regulatory_training_data(self, specialization: str) -> Dict[str, Any]:
        """Get regulatory training data."""
        try:
            # Simulate regulatory data
            await asyncio.sleep(0.1)
            
            return {
                'documents': [
                    f"SEC regulations for {specialization}",
                    f"Compliance requirements for {specialization}",
                    f"Regulatory framework for {specialization}",
                    f"Audit guidelines for {specialization}"
                ],
                'categories': ['sec', 'compliance', 'framework', 'audit'],
                'risk_levels': ['low', 'medium', 'high', 'medium']
            }
            
        except Exception as e:
            logger.error(f"Error getting regulatory training data: {str(e)}")
            return {}
    
    async def _get_execution_training_data(self, specialization: str) -> Dict[str, Any]:
        """Get execution training data."""
        try:
            # Simulate execution data
            await asyncio.sleep(0.1)
            
            return {
                'execution_times': [0.023, 0.045, 0.012, 0.067, 0.034],
                'success_rates': [0.95, 0.87, 0.98, 0.92, 0.96],
                'slippage': [0.001, 0.002, 0.0005, 0.003, 0.0015],
                'order_sizes': [1000, 5000, 2500, 10000, 3000]
            }
            
        except Exception as e:
            logger.error(f"Error getting execution training data: {str(e)}")
            return {}
    
    async def _get_generic_training_data(self, data_source: str, specialization: str) -> Dict[str, Any]:
        """Get generic training data."""
        try:
            # Simulate generic data
            await asyncio.sleep(0.1)
            
            return {
                'data_type': data_source,
                'specialization': specialization,
                'samples': np.random.rand(100).tolist(),
                'features': [f'feature_{i}' for i in range(10)],
                'labels': np.random.randint(0, 2, 100).tolist()
            }
            
        except Exception as e:
            logger.error(f"Error getting generic training data: {str(e)}")
            return {}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache or cache_key not in self.cache_timestamps:
            return False
        
        timestamp = self.cache_timestamps[cache_key]
        age = (datetime.now() - timestamp).total_seconds()
        
        return age < self.cache_ttl
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.now()
    
    def _assess_dataframe_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess quality of DataFrame."""
        try:
            total_rows = len(df)
            total_cols = len(df.columns)
            
            # Check for missing values
            missing_data = df.isnull().sum().to_dict()
            missing_percentage = (df.isnull().sum().sum() / (total_rows * total_cols)) * 100
            
            # Check for duplicates
            duplicate_rows = df.duplicated().sum()
            duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
            
            # Check data types
            data_types = df.dtypes.to_dict()
            
            return {
                'total_rows': total_rows,
                'total_columns': total_cols,
                'missing_data': missing_data,
                'missing_percentage': missing_percentage,
                'duplicate_rows': duplicate_rows,
                'duplicate_percentage': duplicate_percentage,
                'data_types': data_types,
                'quality_score': max(0, 100 - missing_percentage - duplicate_percentage)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _assess_dict_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of dictionary data."""
        try:
            total_keys = len(data)
            empty_values = sum(1 for v in data.values() if v is None or v == "")
            empty_percentage = (empty_values / total_keys) * 100 if total_keys > 0 else 0
            
            return {
                'total_keys': total_keys,
                'empty_values': empty_values,
                'empty_percentage': empty_percentage,
                'quality_score': max(0, 100 - empty_percentage)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame data."""
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning DataFrame: {str(e)}")
            return df
    
    def _clean_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean dictionary data."""
        try:
            # Remove None values
            cleaned = {k: v for k, v in data.items() if v is not None}
            
            # Convert empty strings to None
            for key, value in cleaned.items():
                if isinstance(value, str) and value.strip() == "":
                    cleaned[key] = None
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning dictionary: {str(e)}")
            return data
    
    async def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Data cache cleared")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'cache_keys': list(self.cache.keys()),
            'oldest_entry': min(self.cache_timestamps.values()) if self.cache_timestamps else None,
            'newest_entry': max(self.cache_timestamps.values()) if self.cache_timestamps else None
        } 