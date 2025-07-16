"""
Market Data Provider - Handles market data access, caching, and real-time updates.
Provides unified interface for accessing various market data sources.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import random

logger = logging.getLogger(__name__)

class MarketDataProvider:
    """
    Market Data Provider for accessing and caching market data.
    
    Features:
    - Real-time and historical market data access
    - Data caching and optimization
    - Multiple data source support
    - Data validation and cleaning
    """
    
    def __init__(self):
        """Initialize the Market Data Provider."""
        self.cache = {}
        self.cache_expiry = {}
        self.is_initialized = False
        self.data_sources = {
            "primary": "simulated",
            "backup": "simulated"
        }
        
        # Cache settings
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 1000
        
        logger.info("Market Data Provider initialized")
    
    async def initialize(self) -> bool:
        """Initialize the Market Data Provider."""
        try:
            # Initialize data sources
            await self._initialize_data_sources()
            
            # Set up cache cleanup
            asyncio.create_task(self._cache_cleanup_task())
            
            self.is_initialized = True
            logger.info("Market Data Provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Market Data Provider: {str(e)}")
            return False
    
    async def _initialize_data_sources(self):
        """Initialize data sources."""
        try:
            # In a real implementation, this would connect to actual data providers
            # For demo purposes, we'll use simulated data
            
            logger.info("Initializing simulated data sources")
            
            # Initialize with some sample data
            sample_symbols = ["SPY", "QQQ", "AAPL", "GOOGL", "MSFT", "TSLA", "BTC", "ETH"]
            for symbol in sample_symbols:
                await self._generate_sample_data(symbol)
            
        except Exception as e:
            logger.error(f"Error initializing data sources: {str(e)}")
    
    async def _generate_sample_data(self, symbol: str):
        """Generate sample market data for a symbol."""
        try:
            # Generate realistic price data
            base_price = self._get_base_price(symbol)
            volatility = self._get_volatility(symbol)
            
            # Generate 365 days of price data
            prices = []
            current_price = base_price
            
            for i in range(365):
                # Random walk with drift
                daily_return = random.gauss(0.0005, volatility)  # 0.05% daily drift
                current_price *= (1 + daily_return)
                prices.append(round(current_price, 2))
            
            # Generate volume data
            volumes = [random.randint(1000000, 10000000) for _ in range(365)]
            
            # Store in cache
            self.cache[symbol] = {
                "prices": prices,
                "volumes": volumes,
                "last_updated": datetime.now(),
                "data_points": len(prices)
            }
            
            self.cache_expiry[symbol] = datetime.now() + timedelta(seconds=self.cache_ttl)
            
        except Exception as e:
            logger.error(f"Error generating sample data for {symbol}: {str(e)}")
    
    def _get_base_price(self, symbol: str) -> float:
        """Get base price for a symbol."""
        base_prices = {
            "SPY": 450.0, "QQQ": 380.0, "AAPL": 150.0, "GOOGL": 2800.0,
            "MSFT": 300.0, "TSLA": 200.0, "BTC": 45000.0, "ETH": 3000.0
        }
        return base_prices.get(symbol, 100.0)
    
    def _get_volatility(self, symbol: str) -> float:
        """Get volatility for a symbol."""
        volatilities = {
            "SPY": 0.015, "QQQ": 0.020, "AAPL": 0.025, "GOOGL": 0.025,
            "MSFT": 0.020, "TSLA": 0.040, "BTC": 0.050, "ETH": 0.060
        }
        return volatilities.get(symbol, 0.020)
    
    async def get_historical_data(
        self, 
        symbol: str, 
        days: int = 30,
        interval: str = "1d"
    ) -> Optional[Dict[str, Any]]:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol: Trading symbol
            days: Number of days of data
            interval: Data interval (1d, 1h, 1m)
            
        Returns:
            Historical market data
        """
        try:
            # Check cache first
            if symbol in self.cache:
                cached_data = self.cache[symbol]
                if self._is_cache_valid(symbol):
                    # Return requested number of days
                    if days <= len(cached_data["prices"]):
                        return {
                            "symbol": symbol,
                            "prices": cached_data["prices"][-days:],
                            "volumes": cached_data["volumes"][-days:] if "volumes" in cached_data else [],
                            "last_updated": cached_data["last_updated"],
                            "data_points": days
                        }
            
            # Generate new data if not in cache
            await self._generate_sample_data(symbol)
            
            if symbol in self.cache:
                cached_data = self.cache[symbol]
                return {
                    "symbol": symbol,
                    "prices": cached_data["prices"][-days:],
                    "volumes": cached_data["volumes"][-days:] if "volumes" in cached_data else [],
                    "last_updated": cached_data["last_updated"],
                    "data_points": days
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return None
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get real-time market data for symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Real-time market data
        """
        try:
            real_time_data = {}
            
            for symbol in symbols:
                if symbol in self.cache and self._is_cache_valid(symbol):
                    cached_data = self.cache[symbol]
                    current_price = cached_data["prices"][-1] if cached_data["prices"] else 0
                    
                    # Simulate real-time updates
                    price_change = random.gauss(0, 0.001)  # Small random change
                    current_price *= (1 + price_change)
                    
                    real_time_data[symbol] = {
                        "price": round(current_price, 2),
                        "change": round(price_change * 100, 2),
                        "volume": cached_data["volumes"][-1] if cached_data["volumes"] else 0,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Generate data if not available
                    await self._generate_sample_data(symbol)
                    if symbol in self.cache:
                        cached_data = self.cache[symbol]
                        real_time_data[symbol] = {
                            "price": cached_data["prices"][-1] if cached_data["prices"] else 0,
                            "change": 0.0,
                            "volume": cached_data["volumes"][-1] if cached_data["volumes"] else 0,
                            "timestamp": datetime.now().isoformat()
                        }
            
            return real_time_data
            
        except Exception as e:
            logger.error(f"Error getting real-time data: {str(e)}")
            return {}
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid."""
        if symbol not in self.cache_expiry:
            return False
        
        return datetime.now() < self.cache_expiry[symbol]
    
    async def _cache_cleanup_task(self):
        """Periodic cache cleanup task."""
        while True:
            try:
                # Remove expired entries
                current_time = datetime.now()
                expired_symbols = [
                    symbol for symbol, expiry in self.cache_expiry.items()
                    if current_time > expiry
                ]
                
                for symbol in expired_symbols:
                    del self.cache[symbol]
                    del self.cache_expiry[symbol]
                
                if expired_symbols:
                    logger.info(f"Cleaned up {len(expired_symbols)} expired cache entries")
                
                # Limit cache size
                if len(self.cache) > self.max_cache_size:
                    # Remove oldest entries
                    sorted_symbols = sorted(
                        self.cache.keys(),
                        key=lambda x: self.cache[x].get("last_updated", datetime.min)
                    )
                    
                    symbols_to_remove = sorted_symbols[:-self.max_cache_size]
                    for symbol in symbols_to_remove:
                        del self.cache[symbol]
                        if symbol in self.cache_expiry:
                            del self.cache_expiry[symbol]
                    
                    logger.info(f"Removed {len(symbols_to_remove)} old cache entries")
                
                await asyncio.sleep(60)  # Clean up every minute
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {str(e)}")
                await asyncio.sleep(60)
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get market summary information."""
        try:
            summary = {
                "total_symbols": len(self.cache),
                "cache_status": {
                    "valid_entries": len([s for s in self.cache.keys() if self._is_cache_valid(s)]),
                    "expired_entries": len([s for s in self.cache.keys() if not self._is_cache_valid(s)])
                },
                "data_sources": self.data_sources,
                "last_updated": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting market summary: {str(e)}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown the Market Data Provider."""
        try:
            logger.info("Shutting down Market Data Provider")
            
            # Clear cache
            self.cache.clear()
            self.cache_expiry.clear()
            
            self.is_initialized = False
            
        except Exception as e:
            logger.error(f"Error during Market Data Provider shutdown: {str(e)}") 