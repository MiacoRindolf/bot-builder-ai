"""
Risk Manager AI Employee - Specialized in risk assessment and portfolio risk management.
Handles risk analysis, position monitoring, compliance checks, and risk reporting.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import uuid

from employees.base_employee import BaseAIEmployee
from config.settings import settings
from utils.risk_calculator import RiskCalculator
from utils.portfolio_analyzer import PortfolioAnalyzer
from utils.compliance_checker import ComplianceChecker

logger = logging.getLogger(__name__)

class RiskManager(BaseAIEmployee):
    """
    Risk Manager AI Employee specialized in risk assessment and management.
    
    Capabilities:
    - Portfolio risk analysis and monitoring
    - Position risk assessment
    - Compliance monitoring and reporting
    - Risk limit enforcement
    - Stress testing and scenario analysis
    """
    
    def __init__(self, employee_id: str, spec: Any, context: Any = None):
        """Initialize the Risk Manager AI Employee."""
        super().__init__(employee_id, spec, context)
        
        self.role = "risk_manager"
        self.specialization = spec.specialization if hasattr(spec, 'specialization') else "general"
        
        # Risk management attributes
        self.risk_limits = {
            "max_portfolio_risk": 0.15,      # 15% maximum portfolio risk
            "max_position_risk": 0.05,       # 5% maximum position risk
            "max_correlation": 0.7,          # Maximum correlation between positions
            "max_leverage": 2.0,             # Maximum leverage ratio
            "max_drawdown": 0.20,            # 20% maximum drawdown
            "var_limit": 0.02                # 2% Value at Risk limit
        }
        
        # Risk analysis tools
        self.risk_calculator = RiskCalculator()
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.compliance_checker = ComplianceChecker()
        
        # Risk monitoring state
        self.portfolio_risk = 0.0
        self.position_risks = {}
        self.risk_alerts = []
        self.compliance_violations = []
        
        logger.info(f"Risk Manager AI Employee {employee_id} initialized with specialization: {self.specialization}")
    
    async def initialize(self) -> bool:
        """Initialize the Risk Manager AI Employee."""
        try:
            # Initialize base employee
            if not await super().initialize():
                return False
            
            # Initialize risk analysis tools
            await self.risk_calculator.initialize()
            await self.portfolio_analyzer.initialize()
            await self.compliance_checker.initialize()
            
            # Set up continuous risk monitoring
            await self._setup_risk_monitoring()
            
            logger.info(f"Risk Manager AI Employee {self.employee_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Risk Manager AI Employee {self.employee_id}: {str(e)}")
            return False
    
    async def _setup_risk_monitoring(self):
        """Set up continuous risk monitoring."""
        try:
            # Schedule risk monitoring tasks
            asyncio.create_task(self._continuous_risk_monitoring())
            asyncio.create_task(self._compliance_monitoring())
            asyncio.create_task(self._risk_reporting())
            
            logger.info("Risk monitoring setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up risk monitoring: {str(e)}")
    
    async def assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk assessment.
        
        Args:
            portfolio_data: Portfolio positions and market data
            
        Returns:
            Detailed risk assessment results
        """
        try:
            risk_assessment = {
                "timestamp": datetime.now().isoformat(),
                "portfolio_risk": 0.0,
                "position_risks": {},
                "correlation_analysis": {},
                "var_analysis": {},
                "stress_test_results": {},
                "risk_alerts": [],
                "recommendations": []
            }
            
            # Calculate individual position risks
            for symbol, position in portfolio_data.get("positions", {}).items():
                position_risk = await self._calculate_position_risk(symbol, position)
                risk_assessment["position_risks"][symbol] = position_risk
            
            # Calculate portfolio-level risk
            portfolio_risk = await self._calculate_portfolio_risk(risk_assessment["position_risks"])
            risk_assessment["portfolio_risk"] = portfolio_risk
            
            # Correlation analysis
            correlation_analysis = await self._analyze_correlations(portfolio_data)
            risk_assessment["correlation_analysis"] = correlation_analysis
            
            # Value at Risk analysis
            var_analysis = await self._calculate_var(portfolio_data)
            risk_assessment["var_analysis"] = var_analysis
            
            # Stress testing
            stress_results = await self._perform_stress_tests(portfolio_data)
            risk_assessment["stress_test_results"] = stress_results
            
            # Generate risk alerts
            risk_assessment["risk_alerts"] = await self._generate_risk_alerts(risk_assessment)
            
            # Generate recommendations
            risk_assessment["recommendations"] = await self._generate_risk_recommendations(risk_assessment)
            
            # Update internal state
            self.portfolio_risk = portfolio_risk
            self.position_risks = risk_assessment["position_risks"]
            
            logger.info(f"Portfolio risk assessment completed. Overall risk: {portfolio_risk:.2%}")
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error in portfolio risk assessment: {str(e)}")
            return {"error": str(e)}
    
    async def _calculate_position_risk(self, symbol: str, position: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk for an individual position."""
        try:
            quantity = position.get("quantity", 0)
            current_price = position.get("current_price", 0)
            position_value = quantity * current_price
            
            # Market risk (price volatility)
            volatility = position.get("volatility", 0.2)  # Default 20% volatility
            market_risk = volatility * (position_value ** 0.5)
            
            # Liquidity risk
            liquidity_risk = self._assess_liquidity_risk(symbol, quantity)
            
            # Concentration risk
            portfolio_value = position.get("portfolio_value", 1)
            concentration_risk = position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Total position risk
            total_risk = (market_risk + liquidity_risk + concentration_risk) / position_value if position_value > 0 else 0
            
            return {
                "symbol": symbol,
                "position_value": position_value,
                "market_risk": market_risk,
                "liquidity_risk": liquidity_risk,
                "concentration_risk": concentration_risk,
                "total_risk": total_risk,
                "risk_score": min(total_risk, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating position risk for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    def _assess_liquidity_risk(self, symbol: str, quantity: float) -> float:
        """Assess liquidity risk for a position."""
        try:
            # Simplified liquidity risk assessment
            # In a real implementation, this would use bid-ask spreads, volume data, etc.
            
            # High liquidity symbols
            high_liquidity = ["SPY", "QQQ", "AAPL", "GOOGL", "MSFT", "BTC", "ETH"]
            # Medium liquidity symbols
            medium_liquidity = ["IWM", "GLD", "TLT", "TSLA", "AMZN"]
            
            if symbol in high_liquidity:
                return quantity * 0.001  # 0.1% liquidity cost
            elif symbol in medium_liquidity:
                return quantity * 0.005  # 0.5% liquidity cost
            else:
                return quantity * 0.02   # 2% liquidity cost for low liquidity
            
        except Exception as e:
            logger.error(f"Error assessing liquidity risk: {str(e)}")
            return 0.0
    
    async def _calculate_portfolio_risk(self, position_risks: Dict[str, Any]) -> float:
        """Calculate overall portfolio risk."""
        try:
            total_risk = 0.0
            total_value = 0.0
            
            for symbol, risk_data in position_risks.items():
                if "error" in risk_data:
                    continue
                
                position_value = risk_data.get("position_value", 0)
                risk_score = risk_data.get("risk_score", 0)
                
                total_value += position_value
                total_risk += position_value * risk_score
            
            # Normalize by total portfolio value
            portfolio_risk = total_risk / total_value if total_value > 0 else 0.0
            
            return min(portfolio_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            return 0.0
    
    async def _analyze_correlations(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations between portfolio positions."""
        try:
            positions = portfolio_data.get("positions", {})
            symbols = list(positions.keys())
            
            correlation_matrix = {}
            high_correlations = []
            
            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        correlation_matrix[symbol1][symbol2] = 1.0
                    else:
                        # Simplified correlation calculation
                        # In a real implementation, this would use historical price data
                        correlation = self._calculate_correlation(symbol1, symbol2)
                        correlation_matrix[symbol1][symbol2] = correlation
                        
                        if abs(correlation) > self.risk_limits["max_correlation"]:
                            high_correlations.append({
                                "symbol1": symbol1,
                                "symbol2": symbol2,
                                "correlation": correlation
                            })
            
            return {
                "correlation_matrix": correlation_matrix,
                "high_correlations": high_correlations,
                "diversification_score": self._calculate_diversification_score(correlation_matrix)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols (simplified)."""
        try:
            # Simplified correlation calculation for demo
            # In a real implementation, this would use historical returns
            
            # Asset class correlations
            asset_classes = {
                "SPY": "equity", "QQQ": "equity", "IWM": "equity",
                "AAPL": "equity", "GOOGL": "equity", "MSFT": "equity",
                "GLD": "commodity", "TLT": "bond", "BTC": "crypto", "ETH": "crypto"
            }
            
            class1 = asset_classes.get(symbol1, "other")
            class2 = asset_classes.get(symbol2, "other")
            
            # Correlation matrix for asset classes
            correlations = {
                ("equity", "equity"): 0.8,
                ("equity", "bond"): -0.2,
                ("equity", "commodity"): 0.1,
                ("equity", "crypto"): 0.3,
                ("bond", "commodity"): 0.0,
                ("bond", "crypto"): 0.1,
                ("commodity", "crypto"): 0.2
            }
            
            # Get correlation, defaulting to 0 for unknown combinations
            key = tuple(sorted([class1, class2]))
            return correlations.get(key, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return 0.0
    
    def _calculate_diversification_score(self, correlation_matrix: Dict[str, Any]) -> float:
        """Calculate portfolio diversification score."""
        try:
            if not correlation_matrix:
                return 0.0
            
            total_correlation = 0.0
            count = 0
            
            for symbol1, correlations in correlation_matrix.items():
                for symbol2, correlation in correlations.items():
                    if symbol1 != symbol2:
                        total_correlation += abs(correlation)
                        count += 1
            
            avg_correlation = total_correlation / count if count > 0 else 0.0
            diversification_score = 1.0 - avg_correlation
            
            return max(0.0, min(1.0, diversification_score))
            
        except Exception as e:
            logger.error(f"Error calculating diversification score: {str(e)}")
            return 0.0
    
    async def _calculate_var(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR) for the portfolio."""
        try:
            positions = portfolio_data.get("positions", {})
            total_value = sum(pos.get("position_value", 0) for pos in positions.values())
            
            if total_value == 0:
                return {"var_95": 0, "var_99": 0, "expected_shortfall": 0}
            
            # Simplified VaR calculation
            # In a real implementation, this would use historical simulation or Monte Carlo
            
            portfolio_volatility = 0.15  # 15% annual volatility
            confidence_levels = [0.95, 0.99]
            
            var_results = {}
            for confidence in confidence_levels:
                # Normal distribution assumption
                z_score = 1.645 if confidence == 0.95 else 2.326
                var = total_value * portfolio_volatility * z_score / (252 ** 0.5)  # Daily VaR
                var_results[f"var_{int(confidence*100)}"] = var
            
            # Expected shortfall (Conditional VaR)
            expected_shortfall = total_value * portfolio_volatility * 2.063 / (252 ** 0.5)
            var_results["expected_shortfall"] = expected_shortfall
            
            return var_results
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return {"error": str(e)}
    
    async def _perform_stress_tests(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform stress tests on the portfolio."""
        try:
            positions = portfolio_data.get("positions", {})
            total_value = sum(pos.get("position_value", 0) for pos in positions.values())
            
            stress_scenarios = {
                "market_crash": {
                    "equity_shock": -0.20,  # 20% equity decline
                    "bond_shock": 0.05,     # 5% bond decline
                    "commodity_shock": -0.10, # 10% commodity decline
                    "crypto_shock": -0.30    # 30% crypto decline
                },
                "interest_rate_spike": {
                    "equity_shock": -0.10,
                    "bond_shock": -0.15,
                    "commodity_shock": 0.05,
                    "crypto_shock": -0.15
                },
                "inflation_surge": {
                    "equity_shock": -0.05,
                    "bond_shock": -0.20,
                    "commodity_shock": 0.15,
                    "crypto_shock": 0.10
                }
            }
            
            results = {}
            for scenario, shocks in stress_scenarios.items():
                scenario_loss = 0.0
                
                for symbol, position in positions.items():
                    position_value = position.get("position_value", 0)
                    
                    # Determine asset class and apply appropriate shock
                    asset_class = self._get_asset_class(symbol)
                    shock = shocks.get(f"{asset_class}_shock", 0.0)
                    
                    scenario_loss += position_value * shock
                
                results[scenario] = {
                    "loss": scenario_loss,
                    "loss_percentage": scenario_loss / total_value if total_value > 0 else 0
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing stress tests: {str(e)}")
            return {"error": str(e)}
    
    def _get_asset_class(self, symbol: str) -> str:
        """Determine asset class for a symbol."""
        equity_symbols = ["SPY", "QQQ", "IWM", "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        bond_symbols = ["TLT", "IEF", "SHY"]
        commodity_symbols = ["GLD", "SLV", "USO"]
        crypto_symbols = ["BTC", "ETH", "ADA", "DOT"]
        
        if symbol in equity_symbols:
            return "equity"
        elif symbol in bond_symbols:
            return "bond"
        elif symbol in commodity_symbols:
            return "commodity"
        elif symbol in crypto_symbols:
            return "crypto"
        else:
            return "other"
    
    async def _generate_risk_alerts(self, risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk alerts based on assessment."""
        try:
            alerts = []
            
            # Portfolio risk alert
            portfolio_risk = risk_assessment.get("portfolio_risk", 0)
            if portfolio_risk > self.risk_limits["max_portfolio_risk"]:
                alerts.append({
                    "type": "high_portfolio_risk",
                    "severity": "high",
                    "message": f"Portfolio risk {portfolio_risk:.1%} exceeds limit {self.risk_limits['max_portfolio_risk']:.1%}",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Position risk alerts
            for symbol, risk_data in risk_assessment.get("position_risks", {}).items():
                if "error" in risk_data:
                    continue
                
                risk_score = risk_data.get("risk_score", 0)
                if risk_score > self.risk_limits["max_position_risk"]:
                    alerts.append({
                        "type": "high_position_risk",
                        "severity": "medium",
                        "message": f"Position risk for {symbol} {risk_score:.1%} exceeds limit {self.risk_limits['max_position_risk']:.1%}",
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat()
                    })
            
            # VaR alerts
            var_analysis = risk_assessment.get("var_analysis", {})
            var_95 = var_analysis.get("var_95", 0)
            portfolio_value = sum(pos.get("position_value", 0) for pos in risk_assessment.get("position_risks", {}).values())
            
            if var_95 > portfolio_value * self.risk_limits["var_limit"]:
                alerts.append({
                    "type": "high_var",
                    "severity": "high",
                    "message": f"VaR {var_95:.2f} exceeds limit {portfolio_value * self.risk_limits['var_limit']:.2f}",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Correlation alerts
            high_correlations = risk_assessment.get("correlation_analysis", {}).get("high_correlations", [])
            for correlation in high_correlations:
                alerts.append({
                    "type": "high_correlation",
                    "severity": "medium",
                    "message": f"High correlation {correlation['correlation']:.2f} between {correlation['symbol1']} and {correlation['symbol2']}",
                    "symbols": [correlation['symbol1'], correlation['symbol2']],
                    "timestamp": datetime.now().isoformat()
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating risk alerts: {str(e)}")
            return []
    
    async def _generate_risk_recommendations(self, risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk management recommendations."""
        try:
            recommendations = []
            
            # High portfolio risk recommendation
            portfolio_risk = risk_assessment.get("portfolio_risk", 0)
            if portfolio_risk > self.risk_limits["max_portfolio_risk"]:
                recommendations.append({
                    "type": "reduce_portfolio_risk",
                    "priority": "high",
                    "action": "Consider reducing position sizes or adding hedging positions",
                    "rationale": f"Portfolio risk {portfolio_risk:.1%} exceeds target {self.risk_limits['max_portfolio_risk']:.1%}"
                })
            
            # Diversification recommendation
            diversification_score = risk_assessment.get("correlation_analysis", {}).get("diversification_score", 0)
            if diversification_score < 0.5:
                recommendations.append({
                    "type": "improve_diversification",
                    "priority": "medium",
                    "action": "Add positions in different asset classes to improve diversification",
                    "rationale": f"Low diversification score {diversification_score:.2f}"
                })
            
            # High correlation recommendation
            high_correlations = risk_assessment.get("correlation_analysis", {}).get("high_correlations", [])
            if len(high_correlations) > 3:
                recommendations.append({
                    "type": "reduce_correlations",
                    "priority": "medium",
                    "action": "Consider reducing positions with high correlations",
                    "rationale": f"Found {len(high_correlations)} highly correlated position pairs"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating risk recommendations: {str(e)}")
            return []
    
    async def _continuous_risk_monitoring(self):
        """Continuous risk monitoring task."""
        while self.is_running:
            try:
                # This would typically monitor real-time portfolio data
                # For demo purposes, we'll simulate periodic monitoring
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in continuous risk monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def _compliance_monitoring(self):
        """Compliance monitoring task."""
        while self.is_running:
            try:
                # Check compliance rules
                compliance_check = await self.compliance_checker.check_compliance()
                
                if compliance_check.get("violations"):
                    self.compliance_violations.extend(compliance_check["violations"])
                    logger.warning(f"Compliance violations detected: {len(compliance_check['violations'])}")
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in compliance monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def _risk_reporting(self):
        """Periodic risk reporting task."""
        while self.is_running:
            try:
                # Generate daily risk report
                risk_report = await self.generate_risk_report()
                
                # Log risk summary
                logger.info(f"Daily risk report - Portfolio risk: {risk_report.get('portfolio_risk', 0):.2%}, "
                          f"Alerts: {len(risk_report.get('alerts', []))}")
                
                await asyncio.sleep(86400)  # Daily report
                
            except Exception as e:
                logger.error(f"Error in risk reporting: {str(e)}")
                await asyncio.sleep(3600)
    
    async def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "portfolio_risk": self.portfolio_risk,
                "position_risks": self.position_risks,
                "alerts": self.risk_alerts,
                "compliance_violations": self.compliance_violations,
                "risk_limits": self.risk_limits,
                "summary": {
                    "high_risk_positions": len([r for r in self.position_risks.values() 
                                              if r.get("risk_score", 0) > self.risk_limits["max_position_risk"]]),
                    "active_alerts": len(self.risk_alerts),
                    "compliance_issues": len(self.compliance_violations)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {str(e)}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown the Risk Manager AI Employee."""
        try:
            logger.info(f"Shutting down Risk Manager AI Employee {self.employee_id}")
            
            # Stop background tasks
            self.is_running = False
            
            # Generate final risk report
            final_report = await self.generate_risk_report()
            logger.info(f"Final risk report - Portfolio risk: {final_report.get('portfolio_risk', 0):.2%}")
            
            await super().shutdown()
            
        except Exception as e:
            logger.error(f"Error during Risk Manager shutdown: {str(e)}") 