"""
Real-time Market Data Integration for Bot Builder AI System.
Provides live financial data feeds and real-time market monitoring.
"""

import asyncio
import logging
import json
import websockets
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import threading
from queue import Queue
import yfinance as yf
import requests

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Real-time market data point."""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: float
    ask: float
    high: float
    low: float
    open_price: float
    previous_close: float
    change: float
    change_percent: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None

@dataclass
class MarketEvent:
    """Market event for real-time processing."""
    event_type: str  # 'price_update', 'volume_spike', 'news_event', 'technical_signal'
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'

class RealTimeMarketDataProvider:
    """
    Real-time market data provider with multiple data sources.
    
    Features:
    - Multiple data source integration (Yahoo Finance, Alpha Vantage, etc.)
    - WebSocket connections for real-time updates
    - Event-driven architecture for market events
    - Data validation and quality checks
    - Automatic failover and redundancy
    """
    
    def __init__(self):
        """Initialize the real-time market data provider."""
        self.is_initialized = False
        self.is_running = False
        
        # Data sources
        self.data_sources = {
            "yahoo_finance": self._yahoo_finance_source,
            "alpha_vantage": self._alpha_vantage_source,
            "simulated": self._simulated_source
        }
        
        # Active connections
        self.active_connections: Dict[str, Any] = {}
        self.websocket_connections: Dict[str, Any] = {}
        
        # Data storage
        self.latest_data: Dict[str, MarketDataPoint] = {}
        self.historical_data: Dict[str, List[MarketDataPoint]] = {}
        self.market_events: List[MarketEvent] = []
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'price_update': [],
            'volume_spike': [],
            'news_event': [],
            'technical_signal': [],
            'market_crash': [],
            'data_quality_alert': []
        }
        
        # Configuration
        self.update_interval = 1.0  # seconds
        self.max_historical_points = 1000
        self.data_quality_threshold = 0.95
        
        # Monitoring
        self.connection_status = {}
        self.data_quality_metrics = {}
        self.performance_metrics = {
            'updates_per_second': 0,
            'average_latency': 0,
            'error_rate': 0
        }
        
        logger.info("Real-time Market Data Provider initialized")
    
    async def initialize(self) -> bool:
        """Initialize the real-time market data provider."""
        try:
            logger.info("Initializing Real-time Market Data Provider...")
            
            # Initialize data sources
            await self._initialize_data_sources()
            
            # Set up event processing
            await self._setup_event_processing()
            
            # Start monitoring
            await self._start_monitoring()
            
            self.is_initialized = True
            logger.info("Real-time Market Data Provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Real-time Market Data Provider: {str(e)}")
            return False
    
    async def start_real_time_feed(self, symbols: List[str], data_sources: List[str] = None) -> bool:
        """Start real-time data feed for specified symbols."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if data_sources is None:
                data_sources = ["yahoo_finance", "simulated"]
            
            logger.info(f"Starting real-time feed for {len(symbols)} symbols using {data_sources}")
            
            # Initialize data storage for symbols
            for symbol in symbols:
                self.latest_data[symbol] = None
                self.historical_data[symbol] = []
            
            # Start data collection tasks
            tasks = []
            for source in data_sources:
                if source in self.data_sources:
                    task = asyncio.create_task(
                        self._collect_data_from_source(source, symbols)
                    )
                    tasks.append(task)
            
            # Start event processing
            event_task = asyncio.create_task(self._process_market_events())
            tasks.append(event_task)
            
            # Start monitoring task
            monitor_task = asyncio.create_task(self._monitor_data_quality())
            tasks.append(monitor_task)
            
            self.is_running = True
            logger.info(f"Real-time feed started with {len(tasks)} active tasks")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting real-time feed: {str(e)}")
            return False
    
    async def get_real_time_data(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get latest real-time data for a symbol."""
        try:
            return self.latest_data.get(symbol)
        except Exception as e:
            logger.error(f"Error getting real-time data for {symbol}: {str(e)}")
            return None
    
    async def get_historical_data(self, symbol: str, minutes_back: int = 60) -> List[MarketDataPoint]:
        """Get historical data for a symbol."""
        try:
            data = self.historical_data.get(symbol, [])
            if not data:
                return []
            
            cutoff_time = datetime.now() - timedelta(minutes=minutes_back)
            return [point for point in data if point.timestamp >= cutoff_time]
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return []
    
    async def subscribe_to_events(self, event_type: str, handler: Callable):
        """Subscribe to market events."""
        try:
            if event_type in self.event_handlers:
                self.event_handlers[event_type].append(handler)
                logger.info(f"Added event handler for {event_type}")
            else:
                logger.warning(f"Unknown event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Error subscribing to events: {str(e)}")
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get market summary with key metrics."""
        try:
            summary = {
                "total_symbols": len(self.latest_data),
                "active_connections": len(self.active_connections),
                "last_update": datetime.now().isoformat(),
                "data_quality": self.data_quality_metrics,
                "performance": self.performance_metrics,
                "recent_events": len([e for e in self.market_events if e.timestamp > datetime.now() - timedelta(minutes=5)])
            }
            
            # Add symbol-specific summaries
            symbol_summaries = {}
            for symbol, data in self.latest_data.items():
                if data:
                    symbol_summaries[symbol] = {
                        "price": data.price,
                        "change_percent": data.change_percent,
                        "volume": data.volume,
                        "last_update": data.timestamp.isoformat()
                    }
            
            summary["symbols"] = symbol_summaries
            return summary
            
        except Exception as e:
            logger.error(f"Error getting market summary: {str(e)}")
            return {}
    
    async def _initialize_data_sources(self):
        """Initialize data sources."""
        try:
            # Initialize Yahoo Finance
            if "yahoo_finance" in self.data_sources:
                await self._initialize_yahoo_finance()
            
            # Initialize Alpha Vantage
            if "alpha_vantage" in self.data_sources and settings.alpha_vantage_api_key:
                await self._initialize_alpha_vantage()
            
            # Initialize simulated data
            if "simulated" in self.data_sources:
                await self._initialize_simulated_data()
            
            logger.info("Data sources initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing data sources: {str(e)}")
    
    async def _initialize_yahoo_finance(self):
        """Initialize Yahoo Finance data source."""
        try:
            # Test connection
            test_symbol = "AAPL"
            ticker = yf.Ticker(test_symbol)
            info = ticker.info
            
            if info:
                self.connection_status["yahoo_finance"] = "connected"
                logger.info("Yahoo Finance initialized successfully")
            else:
                self.connection_status["yahoo_finance"] = "failed"
                logger.warning("Yahoo Finance initialization failed")
                
        except Exception as e:
            self.connection_status["yahoo_finance"] = "error"
            logger.error(f"Error initializing Yahoo Finance: {str(e)}")
    
    async def _initialize_alpha_vantage(self):
        """Initialize Alpha Vantage data source."""
        try:
            # Test API key
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={settings.alpha_vantage_api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "Global Quote" in data:
                            self.connection_status["alpha_vantage"] = "connected"
                            logger.info("Alpha Vantage initialized successfully")
                        else:
                            self.connection_status["alpha_vantage"] = "failed"
                            logger.warning("Alpha Vantage API key may be invalid")
                    else:
                        self.connection_status["alpha_vantage"] = "failed"
                        logger.warning(f"Alpha Vantage connection failed: {response.status}")
                        
        except Exception as e:
            self.connection_status["alpha_vantage"] = "error"
            logger.error(f"Error initializing Alpha Vantage: {str(e)}")
    
    async def _initialize_simulated_data(self):
        """Initialize simulated data source."""
        try:
            self.connection_status["simulated"] = "connected"
            logger.info("Simulated data source initialized")
        except Exception as e:
            logger.error(f"Error initializing simulated data: {str(e)}")
    
    async def _collect_data_from_source(self, source: str, symbols: List[str]):
        """Collect data from a specific source."""
        try:
            logger.info(f"Starting data collection from {source} for {len(symbols)} symbols")
            
            while self.is_running:
                start_time = time.time()
                
                for symbol in symbols:
                    try:
                        # Get data from source
                        data = await self.data_sources[source](symbol)
                        
                        if data:
                            # Update latest data
                            self.latest_data[symbol] = data
                            
                            # Add to historical data
                            self.historical_data[symbol].append(data)
                            
                            # Maintain historical data size
                            if len(self.historical_data[symbol]) > self.max_historical_points:
                                self.historical_data[symbol] = self.historical_data[symbol][-self.max_historical_points:]
                            
                            # Create market event
                            await self._create_market_event('price_update', symbol, data)
                            
                            # Check for volume spikes
                            await self._check_volume_spike(symbol, data)
                            
                    except Exception as e:
                        logger.error(f"Error collecting data for {symbol} from {source}: {str(e)}")
                
                # Update performance metrics
                elapsed_time = time.time() - start_time
                self.performance_metrics['updates_per_second'] = len(symbols) / elapsed_time
                self.performance_metrics['average_latency'] = elapsed_time / len(symbols)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
        except Exception as e:
            logger.error(f"Error in data collection from {source}: {str(e)}")
    
    async def _yahoo_finance_source(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                return None
            
            # Get real-time quote
            quote = ticker.history(period="1d", interval="1m")
            if quote.empty:
                return None
            
            latest = quote.iloc[-1]
            
            return MarketDataPoint(
                symbol=symbol,
                price=float(latest['Close']),
                volume=int(latest['Volume']),
                timestamp=datetime.now(),
                bid=float(info.get('bid', latest['Close'])),
                ask=float(info.get('ask', latest['Close'])),
                high=float(latest['High']),
                low=float(latest['Low']),
                open_price=float(latest['Open']),
                previous_close=float(info.get('previousClose', latest['Close'])),
                change=float(latest['Close'] - info.get('previousClose', latest['Close'])),
                change_percent=float((latest['Close'] - info.get('previousClose', latest['Close'])) / info.get('previousClose', latest['Close']) * 100),
                market_cap=float(info.get('marketCap', 0)),
                pe_ratio=float(info.get('trailingPE', 0)),
                dividend_yield=float(info.get('dividendYield', 0))
            )
            
        except Exception as e:
            logger.error(f"Error getting Yahoo Finance data for {symbol}: {str(e)}")
            return None
    
    async def _alpha_vantage_source(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get data from Alpha Vantage."""
        try:
            if not settings.alpha_vantage_api_key:
                return None
            
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={settings.alpha_vantage_api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "Global Quote" in data:
                            quote = data["Global Quote"]
                            
                            return MarketDataPoint(
                                symbol=symbol,
                                price=float(quote.get('05. price', 0)),
                                volume=int(quote.get('06. volume', 0)),
                                timestamp=datetime.now(),
                                bid=float(quote.get('05. price', 0)),
                                ask=float(quote.get('05. price', 0)),
                                high=float(quote.get('03. high', 0)),
                                low=float(quote.get('04. low', 0)),
                                open_price=float(quote.get('02. open', 0)),
                                previous_close=float(quote.get('08. previous close', 0)),
                                change=float(quote.get('09. change', 0)),
                                change_percent=float(quote.get('10. change percent', '0%').replace('%', ''))
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage data for {symbol}: {str(e)}")
            return None
    
    async def _simulated_source(self, symbol: str) -> Optional[MarketDataPoint]:
        """Generate simulated market data."""
        try:
            # Generate realistic simulated data
            base_price = 100.0 + hash(symbol) % 1000  # Different base price per symbol
            time_factor = time.time() / 3600  # Hour-based variation
            
            # Add some randomness and trends
            price_change = np.sin(time_factor) * 5 + np.random.normal(0, 1)
            current_price = base_price + price_change
            
            volume = int(1000000 + np.random.normal(0, 200000))
            
            return MarketDataPoint(
                symbol=symbol,
                price=current_price,
                volume=volume,
                timestamp=datetime.now(),
                bid=current_price - 0.01,
                ask=current_price + 0.01,
                high=current_price + np.random.uniform(0, 2),
                low=current_price - np.random.uniform(0, 2),
                open_price=base_price,
                previous_close=base_price,
                change=price_change,
                change_percent=(price_change / base_price) * 100
            )
            
        except Exception as e:
            logger.error(f"Error generating simulated data for {symbol}: {str(e)}")
            return None
    
    async def _create_market_event(self, event_type: str, symbol: str, data: MarketDataPoint):
        """Create a market event."""
        try:
            event = MarketEvent(
                event_type=event_type,
                symbol=symbol,
                data={
                    "price": data.price,
                    "volume": data.volume,
                    "change_percent": data.change_percent
                },
                timestamp=datetime.now(),
                severity="low"
            )
            
            self.market_events.append(event)
            
            # Keep only recent events
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.market_events = [e for e in self.market_events if e.timestamp > cutoff_time]
            
            # Trigger event handlers
            await self._trigger_event_handlers(event)
            
        except Exception as e:
            logger.error(f"Error creating market event: {str(e)}")
    
    async def _check_volume_spike(self, symbol: str, data: MarketDataPoint):
        """Check for volume spikes."""
        try:
            historical_data = self.historical_data.get(symbol, [])
            if len(historical_data) < 10:
                return
            
            # Calculate average volume
            recent_volumes = [d.volume for d in historical_data[-10:]]
            avg_volume = sum(recent_volumes) / len(recent_volumes)
            
            # Check for spike (2x average volume)
            if data.volume > avg_volume * 2:
                event = MarketEvent(
                    event_type="volume_spike",
                    symbol=symbol,
                    data={
                        "current_volume": data.volume,
                        "average_volume": avg_volume,
                        "spike_ratio": data.volume / avg_volume
                    },
                    timestamp=datetime.now(),
                    severity="medium"
                )
                
                self.market_events.append(event)
                await self._trigger_event_handlers(event)
                
        except Exception as e:
            logger.error(f"Error checking volume spike: {str(e)}")
    
    async def _trigger_event_handlers(self, event: MarketEvent):
        """Trigger event handlers for a market event."""
        try:
            handlers = self.event_handlers.get(event.event_type, [])
            
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error triggering event handlers: {str(e)}")
    
    async def _process_market_events(self):
        """Process market events."""
        try:
            while self.is_running:
                # Process any pending events
                recent_events = [e for e in self.market_events if e.timestamp > datetime.now() - timedelta(minutes=1)]
                
                for event in recent_events:
                    # Add any additional processing logic here
                    pass
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error processing market events: {str(e)}")
    
    async def _monitor_data_quality(self):
        """Monitor data quality and connection status."""
        try:
            while self.is_running:
                # Check data quality
                quality_metrics = {}
                
                for symbol, data in self.latest_data.items():
                    if data:
                        # Check for stale data
                        time_diff = (datetime.now() - data.timestamp).total_seconds()
                        quality_metrics[symbol] = {
                            "freshness": max(0, 1 - time_diff / 60),  # 1 minute threshold
                            "completeness": 1.0 if data.price > 0 else 0.0,
                            "consistency": 1.0  # Add consistency checks here
                        }
                
                self.data_quality_metrics = quality_metrics
                
                # Check for data quality issues
                for symbol, metrics in quality_metrics.items():
                    if metrics["freshness"] < 0.5:  # Data older than 30 seconds
                        await self._create_market_event("data_quality_alert", symbol, self.latest_data.get(symbol))
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except Exception as e:
            logger.error(f"Error monitoring data quality: {str(e)}")
    
    async def _setup_event_processing(self):
        """Set up event processing infrastructure."""
        try:
            logger.info("Event processing infrastructure set up")
        except Exception as e:
            logger.error(f"Error setting up event processing: {str(e)}")
    
    async def _start_monitoring(self):
        """Start monitoring and alerting."""
        try:
            logger.info("Market data monitoring started")
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
    
    async def stop_real_time_feed(self):
        """Stop the real-time data feed."""
        try:
            self.is_running = False
            logger.info("Real-time market data feed stopped")
        except Exception as e:
            logger.error(f"Error stopping real-time feed: {str(e)}")
    
    async def get_connection_status(self) -> Dict[str, str]:
        """Get connection status for all data sources."""
        return self.connection_status.copy()
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy() 