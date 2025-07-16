"""
Trader AI Employee - Specialized in trading operations and portfolio management.
Handles market analysis, trade execution, risk assessment, and portfolio optimization.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import uuid

from employees.base_employee import BaseAIEmployee
from config.settings import settings
from utils.market_data import MarketDataProvider
from utils.risk_calculator import RiskCalculator
from utils.portfolio_optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)

class Trader(BaseAIEmployee):
    """
    Trader AI Employee specialized in trading operations.
    
    Capabilities:
    - Market analysis and trend identification
    - Trade execution and order management
    - Portfolio optimization and rebalancing
    - Risk assessment and position sizing
    - Performance tracking and reporting
    """
    
    def __init__(self, employee_id: str, spec: Any, context: Any = None):
        """Initialize the Trader AI Employee."""
        super().__init__(employee_id, spec, context)
        
        self.role = "trader"
        self.specialization = spec.specialization if hasattr(spec, 'specialization') else "general"
        
        # Trading-specific attributes
        self.portfolio = {}
        self.positions = {}
        self.trade_history = []
        self.risk_limits = {
            "max_position_size": 0.1,  # 10% of portfolio
            "max_daily_loss": 0.02,    # 2% daily loss limit
            "max_correlation": 0.7,    # Maximum correlation between positions
            "stop_loss_threshold": 0.05  # 5% stop loss
        }
        
        # Market data and analysis tools
        self.market_data = MarketDataProvider()
        self.risk_calculator = RiskCalculator()
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
        logger.info(f"Trader AI Employee {employee_id} initialized with specialization: {self.specialization}")
    
    async def initialize(self) -> bool:
        """Initialize the Trader AI Employee."""
        try:
            # Initialize base employee
            if not await super().initialize():
                return False
            
            # Initialize market data provider
            await self.market_data.initialize()
            
            # Load historical data for analysis
            await self._load_market_data()
            
            # Initialize portfolio
            await self._initialize_portfolio()
            
            # Set up monitoring
            await self._setup_performance_monitoring()
            
            logger.info(f"Trader AI Employee {self.employee_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Trader AI Employee {self.employee_id}: {str(e)}")
            return False
    
    async def _load_market_data(self):
        """Load historical market data for analysis."""
        try:
            # Load data based on specialization
            if self.specialization == "crypto":
                symbols = ["BTC", "ETH", "ADA", "DOT", "LINK"]
            elif self.specialization == "forex":
                symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
            elif self.specialization == "stocks":
                symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
            else:
                symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
            
            # Load historical data
            for symbol in symbols:
                data = await self.market_data.get_historical_data(
                    symbol, 
                    days=365,
                    interval="1d"
                )
                if data:
                    self.market_data.cache[symbol] = data
            
            logger.info(f"Loaded market data for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
    
    async def _initialize_portfolio(self):
        """Initialize the trading portfolio."""
        try:
            # Set initial capital (virtual for demo)
            self.portfolio = {
                "cash": 100000.0,  # $100k initial capital
                "positions": {},
                "total_value": 100000.0,
                "last_updated": datetime.now()
            }
            
            logger.info(f"Portfolio initialized with ${self.portfolio['cash']:,.2f} capital")
            
        except Exception as e:
            logger.error(f"Error initializing portfolio: {str(e)}")
    
    async def _setup_performance_monitoring(self):
        """Set up performance monitoring and alerts."""
        try:
            # Schedule daily performance calculation
            asyncio.create_task(self._daily_performance_task())
            
            # Schedule risk monitoring
            asyncio.create_task(self._risk_monitoring_task())
            
            logger.info("Performance monitoring setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up performance monitoring: {str(e)}")
    
    async def analyze_market(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Analyze market conditions and identify trading opportunities.
        
        Args:
            symbols: List of symbols to analyze (default: all available)
            
        Returns:
            Market analysis results
        """
        try:
            if not symbols:
                symbols = list(self.market_data.cache.keys())
            
            analysis_results = {}
            
            for symbol in symbols:
                if symbol not in self.market_data.cache:
                    continue
                
                data = self.market_data.cache[symbol]
                
                # Technical analysis
                technical_analysis = await self._perform_technical_analysis(data)
                
                # Fundamental analysis (if available)
                fundamental_analysis = await self._perform_fundamental_analysis(symbol)
                
                # Risk assessment
                risk_assessment = await self._assess_position_risk(symbol, data)
                
                analysis_results[symbol] = {
                    "technical": technical_analysis,
                    "fundamental": fundamental_analysis,
                    "risk": risk_assessment,
                    "recommendation": self._generate_trading_recommendation(
                        technical_analysis, fundamental_analysis, risk_assessment
                    )
                }
            
            logger.info(f"Market analysis completed for {len(analysis_results)} symbols")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return {"error": str(e)}
    
    async def _perform_technical_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform technical analysis on market data."""
        try:
            if not data or "prices" not in data:
                return {"error": "No price data available"}
            
            prices = data["prices"]
            if len(prices) < 20:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate technical indicators
            sma_20 = sum(prices[-20:]) / 20
            sma_50 = sum(prices[-50:]) / 50 if len(prices) >= 50 else sma_20
            
            current_price = prices[-1]
            
            # RSI calculation (simplified)
            gains = [max(0, prices[i] - prices[i-1]) for i in range(1, len(prices))]
            losses = [max(0, prices[i-1] - prices[i]) for i in range(1, len(prices))]
            
            avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
            avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0
            
            rsi = 100 - (100 / (1 + (avg_gain / avg_loss if avg_loss > 0 else 1)))
            
            # Volatility calculation
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = (sum([r**2 for r in returns[-20:]]) / 20) ** 0.5
            
            return {
                "current_price": current_price,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "trend": "bullish" if sma_20 > sma_50 else "bearish",
                "rsi": rsi,
                "rsi_signal": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral",
                "volatility": volatility,
                "price_change_1d": (current_price - prices[-2]) / prices[-2] if len(prices) > 1 else 0,
                "price_change_5d": (current_price - prices[-6]) / prices[-6] if len(prices) > 5 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            return {"error": str(e)}
    
    async def _perform_fundamental_analysis(self, symbol: str) -> Dict[str, Any]:
        """Perform fundamental analysis (placeholder for demo)."""
        try:
            # In a real implementation, this would fetch company financials,
            # economic indicators, news sentiment, etc.
            
            return {
                "pe_ratio": None,
                "market_cap": None,
                "revenue_growth": None,
                "profit_margin": None,
                "debt_to_equity": None,
                "sentiment_score": 0.0  # Neutral sentiment
            }
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {str(e)}")
            return {"error": str(e)}
    
    async def _assess_position_risk(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for a potential position in the symbol."""
        try:
            if not data or "prices" not in data:
                return {"error": "No price data available"}
            
            prices = data["prices"]
            current_price = prices[-1]
            
            # Calculate Value at Risk (VaR)
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            var_95 = sorted(returns)[int(len(returns) * 0.05)]
            
            # Calculate maximum drawdown
            peak = prices[0]
            max_drawdown = 0
            for price in prices:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Position sizing recommendation
            portfolio_value = self.portfolio["total_value"]
            max_position_value = portfolio_value * self.risk_limits["max_position_size"]
            recommended_shares = max_position_value / current_price
            
            return {
                "var_95": var_95,
                "max_drawdown": max_drawdown,
                "volatility": (sum([r**2 for r in returns[-20:]]) / 20) ** 0.5,
                "recommended_position_size": recommended_shares,
                "risk_score": min(1.0, (abs(var_95) + max_drawdown) / 2)
            }
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            return {"error": str(e)}
    
    def _generate_trading_recommendation(
        self, 
        technical: Dict[str, Any], 
        fundamental: Dict[str, Any], 
        risk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading recommendation based on analysis."""
        try:
            if "error" in technical or "error" in risk:
                return {"action": "hold", "reason": "Insufficient data"}
            
            # Scoring system
            score = 0.0
            reasons = []
            
            # Technical signals
            if technical.get("trend") == "bullish":
                score += 0.3
                reasons.append("Bullish trend")
            elif technical.get("trend") == "bearish":
                score -= 0.3
                reasons.append("Bearish trend")
            
            if technical.get("rsi_signal") == "oversold":
                score += 0.2
                reasons.append("Oversold conditions")
            elif technical.get("rsi_signal") == "overbought":
                score -= 0.2
                reasons.append("Overbought conditions")
            
            # Risk adjustment
            risk_score = risk.get("risk_score", 0.5)
            if risk_score > 0.7:
                score -= 0.3
                reasons.append("High risk")
            elif risk_score < 0.3:
                score += 0.1
                reasons.append("Low risk")
            
            # Generate recommendation
            if score > 0.3:
                action = "buy"
            elif score < -0.3:
                action = "sell"
            else:
                action = "hold"
            
            return {
                "action": action,
                "confidence": min(abs(score), 1.0),
                "reasons": reasons,
                "score": score
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return {"action": "hold", "reason": "Error in analysis"}
    
    async def execute_trade(
        self, 
        symbol: str, 
        action: str, 
        quantity: float,
        order_type: str = "market"
    ) -> Dict[str, Any]:
        """
        Execute a trade order.
        
        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            quantity: Number of shares/units
            order_type: 'market' or 'limit'
            
        Returns:
            Trade execution result
        """
        try:
            # Validate trade parameters
            if action not in ["buy", "sell"]:
                raise ValueError("Action must be 'buy' or 'sell'")
            
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
            
            # Get current market price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                raise ValueError(f"Unable to get current price for {symbol}")
            
            # Calculate trade value
            trade_value = quantity * current_price
            
            # Check risk limits
            risk_check = await self._check_risk_limits(symbol, action, quantity, trade_value)
            if not risk_check["allowed"]:
                return {
                    "success": False,
                    "error": f"Risk limit exceeded: {risk_check['reason']}"
                }
            
            # Check portfolio constraints
            if action == "buy":
                if self.portfolio["cash"] < trade_value:
                    return {
                        "success": False,
                        "error": "Insufficient cash for purchase"
                    }
            else:  # sell
                if symbol not in self.portfolio["positions"] or \
                   self.portfolio["positions"][symbol] < quantity:
                    return {
                        "success": False,
                        "error": "Insufficient shares to sell"
                    }
            
            # Execute trade (simulated)
            trade_id = str(uuid.uuid4())
            trade_record = {
                "trade_id": trade_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": current_price,
                "value": trade_value,
                "timestamp": datetime.now(),
                "status": "executed"
            }
            
            # Update portfolio
            await self._update_portfolio(trade_record)
            
            # Record trade
            self.trade_history.append(trade_record)
            
            # Update performance metrics
            await self._update_performance_metrics(trade_record)
            
            logger.info(f"Trade executed: {action} {quantity} {symbol} at ${current_price:.2f}")
            
            return {
                "success": True,
                "trade_id": trade_id,
                "trade_record": trade_record,
                "portfolio_value": self.portfolio["total_value"]
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        try:
            # In a real implementation, this would fetch live market data
            # For demo, use cached data or generate realistic price
            if symbol in self.market_data.cache and "prices" in self.market_data.cache[symbol]:
                return self.market_data.cache[symbol]["prices"][-1]
            
            # Generate realistic price for demo
            base_prices = {
                "BTC": 45000, "ETH": 3000, "AAPL": 150, "GOOGL": 2800,
                "SPY": 450, "QQQ": 380, "EUR/USD": 1.08, "GBP/USD": 1.26
            }
            
            return base_prices.get(symbol, 100.0)
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    async def _check_risk_limits(
        self, 
        symbol: str, 
        action: str, 
        quantity: float, 
        trade_value: float
    ) -> Dict[str, Any]:
        """Check if trade complies with risk limits."""
        try:
            # Check position size limit
            portfolio_value = self.portfolio["total_value"]
            position_size_ratio = trade_value / portfolio_value
            
            if position_size_ratio > self.risk_limits["max_position_size"]:
                return {
                    "allowed": False,
                    "reason": f"Position size {position_size_ratio:.1%} exceeds limit {self.risk_limits['max_position_size']:.1%}"
                }
            
            # Check daily loss limit
            if self.daily_pnl < -(portfolio_value * self.risk_limits["max_daily_loss"]):
                return {
                    "allowed": False,
                    "reason": "Daily loss limit reached"
                }
            
            return {"allowed": True, "reason": "Trade within risk limits"}
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return {"allowed": False, "reason": f"Error in risk check: {str(e)}"}
    
    async def _update_portfolio(self, trade_record: Dict[str, Any]):
        """Update portfolio after trade execution."""
        try:
            symbol = trade_record["symbol"]
            action = trade_record["action"]
            quantity = trade_record["quantity"]
            value = trade_record["value"]
            
            if action == "buy":
                # Deduct cash
                self.portfolio["cash"] -= value
                
                # Add position
                if symbol not in self.portfolio["positions"]:
                    self.portfolio["positions"][symbol] = 0
                self.portfolio["positions"][symbol] += quantity
                
            else:  # sell
                # Add cash
                self.portfolio["cash"] += value
                
                # Reduce position
                self.portfolio["positions"][symbol] -= quantity
                if self.portfolio["positions"][symbol] <= 0:
                    del self.portfolio["positions"][symbol]
            
            # Update total value
            await self._calculate_portfolio_value()
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {str(e)}")
    
    async def _calculate_portfolio_value(self):
        """Calculate current portfolio total value."""
        try:
            total_value = self.portfolio["cash"]
            
            for symbol, quantity in self.portfolio["positions"].items():
                current_price = await self._get_current_price(symbol)
                if current_price:
                    total_value += quantity * current_price
            
            self.portfolio["total_value"] = total_value
            self.portfolio["last_updated"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {str(e)}")
    
    async def _update_performance_metrics(self, trade_record: Dict[str, Any]):
        """Update performance metrics after trade."""
        try:
            # Calculate trade P&L
            symbol = trade_record["symbol"]
            action = trade_record["action"]
            quantity = trade_record["quantity"]
            price = trade_record["price"]
            
            if action == "sell" and symbol in self.positions:
                # Calculate realized P&L
                avg_cost = self.positions[symbol]["avg_cost"]
                pnl = (price - avg_cost) * quantity
                self.total_pnl += pnl
                self.daily_pnl += pnl
            
            # Update Sharpe ratio (simplified)
            if len(self.trade_history) > 1:
                returns = []
                for i in range(1, len(self.trade_history)):
                    prev_value = self.trade_history[i-1]["value"]
                    curr_value = self.trade_history[i]["value"]
                    returns.append((curr_value - prev_value) / prev_value)
                
                if returns:
                    avg_return = sum(returns) / len(returns)
                    std_return = (sum([(r - avg_return) ** 2 for r in returns]) / len(returns)) ** 0.5
                    self.sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    async def _daily_performance_task(self):
        """Daily performance calculation task."""
        while self.is_running:
            try:
                # Reset daily P&L
                self.daily_pnl = 0.0
                
                # Calculate portfolio value
                await self._calculate_portfolio_value()
                
                # Log daily performance
                logger.info(f"Daily performance - Portfolio: ${self.portfolio['total_value']:,.2f}, "
                          f"Total P&L: ${self.total_pnl:,.2f}, Sharpe: {self.sharpe_ratio:.3f}")
                
                # Wait 24 hours
                await asyncio.sleep(86400)  # 24 hours
                
            except Exception as e:
                logger.error(f"Error in daily performance task: {str(e)}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _risk_monitoring_task(self):
        """Continuous risk monitoring task."""
        while self.is_running:
            try:
                # Check portfolio risk
                portfolio_risk = await self._calculate_portfolio_risk()
                
                # Alert if risk is high
                if portfolio_risk["total_risk"] > 0.7:
                    logger.warning(f"High portfolio risk detected: {portfolio_risk['total_risk']:.2f}")
                
                # Wait 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring task: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _calculate_portfolio_risk(self) -> Dict[str, Any]:
        """Calculate overall portfolio risk."""
        try:
            total_risk = 0.0
            position_risks = {}
            
            for symbol, quantity in self.portfolio["positions"].items():
                if symbol in self.market_data.cache:
                    data = self.market_data.cache[symbol]
                    risk_assessment = await self._assess_position_risk(symbol, data)
                    
                    if "risk_score" in risk_assessment:
                        position_risk = risk_assessment["risk_score"]
                        position_risks[symbol] = position_risk
                        
                        # Weight by position size
                        current_price = await self._get_current_price(symbol)
                        if current_price:
                            position_value = quantity * current_price
                            weight = position_value / self.portfolio["total_value"]
                            total_risk += position_risk * weight
            
            return {
                "total_risk": min(total_risk, 1.0),
                "position_risks": position_risks
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            return {"total_risk": 0.0, "position_risks": {}}
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        try:
            await self._calculate_portfolio_value()
            
            return {
                "total_value": self.portfolio["total_value"],
                "cash": self.portfolio["cash"],
                "positions": self.portfolio["positions"],
                "total_pnl": self.total_pnl,
                "daily_pnl": self.daily_pnl,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "last_updated": self.portfolio["last_updated"].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {str(e)}")
            return {"error": str(e)}
    
    async def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trade history."""
        try:
            return self.trade_history[-limit:] if self.trade_history else []
            
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return []
    
    async def shutdown(self):
        """Shutdown the Trader AI Employee."""
        try:
            logger.info(f"Shutting down Trader AI Employee {self.employee_id}")
            
            # Stop background tasks
            self.is_running = False
            
            # Save final portfolio state
            await self._calculate_portfolio_value()
            
            # Log final performance
            logger.info(f"Final portfolio value: ${self.portfolio['total_value']:,.2f}")
            logger.info(f"Total P&L: ${self.total_pnl:,.2f}")
            logger.info(f"Sharpe ratio: {self.sharpe_ratio:.3f}")
            
            await super().shutdown()
            
        except Exception as e:
            logger.error(f"Error during Trader shutdown: {str(e)}") 