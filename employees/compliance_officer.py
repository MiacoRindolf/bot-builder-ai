"""
Compliance Officer AI Employee - Specialized in regulatory compliance and policy enforcement.
Handles compliance monitoring, policy enforcement, audit preparation, and regulatory reporting.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import uuid

from employees.base_employee import BaseAIEmployee
from config.settings import settings
from utils.compliance_checker import ComplianceChecker
from utils.policy_enforcer import PolicyEnforcer
from utils.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

class ComplianceOfficer(BaseAIEmployee):
    """
    Compliance Officer AI Employee specialized in regulatory compliance.
    
    Capabilities:
    - Regulatory compliance monitoring
    - Policy enforcement and validation
    - Audit trail management
    - Regulatory reporting
    - Compliance training and guidance
    """
    
    def __init__(self, employee_id: str, spec: Any, context: Any = None):
        """Initialize the Compliance Officer AI Employee."""
        super().__init__(employee_id, spec, context)
        
        self.role = "compliance_officer"
        self.specialization = spec.specialization if hasattr(spec, 'specialization') else "general"
        
        # Compliance framework
        self.compliance_policies = {
            "position_limits": {
                "max_single_position": 0.10,  # 10% of portfolio
                "max_sector_exposure": 0.25,  # 25% per sector
                "max_leverage": 2.0,          # 2x leverage limit
                "min_diversification": 5      # Minimum 5 positions
            },
            "trading_restrictions": {
                "restricted_securities": ["PENNY_STOCKS", "CRYPTO_UTILITY_TOKENS"],
                "trading_hours": {"start": "09:30", "end": "16:00"},
                "blackout_periods": ["EARNINGS_ANNOUNCEMENTS", "INSIDER_TRADING"]
            },
            "reporting_requirements": {
                "daily_reports": ["portfolio_summary", "risk_metrics"],
                "weekly_reports": ["compliance_summary", "policy_violations"],
                "monthly_reports": ["regulatory_filing", "audit_summary"]
            }
        }
        
        # Compliance tools
        self.compliance_checker = ComplianceChecker()
        self.policy_enforcer = PolicyEnforcer()
        self.audit_logger = AuditLogger()
        
        # Compliance state
        self.violations = []
        self.audit_trail = []
        self.compliance_reports = []
        
        logger.info(f"Compliance Officer AI Employee {employee_id} initialized with specialization: {self.specialization}")
    
    async def initialize(self) -> bool:
        """Initialize the Compliance Officer AI Employee."""
        try:
            # Initialize base employee
            if not await super().initialize():
                return False
            
            # Initialize compliance tools
            await self.compliance_checker.initialize()
            await self.policy_enforcer.initialize()
            await self.audit_logger.initialize()
            
            # Set up compliance monitoring
            await self._setup_compliance_monitoring()
            
            logger.info(f"Compliance Officer AI Employee {self.employee_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Compliance Officer AI Employee {self.employee_id}: {str(e)}")
            return False
    
    async def _setup_compliance_monitoring(self):
        """Set up continuous compliance monitoring."""
        try:
            # Schedule compliance monitoring tasks
            asyncio.create_task(self._continuous_compliance_monitoring())
            asyncio.create_task(self._policy_enforcement_monitoring())
            asyncio.create_task(self._audit_trail_monitoring())
            asyncio.create_task(self._regulatory_reporting())
            
            logger.info("Compliance monitoring setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up compliance monitoring: {str(e)}")
    
    async def check_compliance(self, portfolio_data: Dict[str, Any], trade_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive compliance check.
        
        Args:
            portfolio_data: Current portfolio state
            trade_data: Proposed trade data (optional)
            
        Returns:
            Compliance check results
        """
        try:
            compliance_results = {
                "timestamp": datetime.now().isoformat(),
                "overall_compliance": True,
                "portfolio_compliance": {},
                "trade_compliance": {},
                "policy_violations": [],
                "recommendations": [],
                "audit_entries": []
            }
            
            # Check portfolio compliance
            portfolio_compliance = await self._check_portfolio_compliance(portfolio_data)
            compliance_results["portfolio_compliance"] = portfolio_compliance
            
            # Check trade compliance if trade data provided
            if trade_data:
                trade_compliance = await self._check_trade_compliance(trade_data, portfolio_data)
                compliance_results["trade_compliance"] = trade_compliance
            
            # Generate policy violations
            violations = await self._identify_policy_violations(compliance_results)
            compliance_results["policy_violations"] = violations
            
            # Update overall compliance status
            compliance_results["overall_compliance"] = len(violations) == 0
            
            # Generate recommendations
            recommendations = await self._generate_compliance_recommendations(compliance_results)
            compliance_results["recommendations"] = recommendations
            
            # Log audit entries
            audit_entries = await self._log_compliance_check(compliance_results)
            compliance_results["audit_entries"] = audit_entries
            
            # Update internal state
            self.violations.extend(violations)
            
            logger.info(f"Compliance check completed. Overall compliance: {compliance_results['overall_compliance']}")
            
            return compliance_results
            
        except Exception as e:
            logger.error(f"Error in compliance check: {str(e)}")
            return {"error": str(e)}
    
    async def _check_portfolio_compliance(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check portfolio-level compliance."""
        try:
            portfolio_compliance = {
                "position_limits": {},
                "diversification": {},
                "leverage": {},
                "restricted_securities": {}
            }
            
            positions = portfolio_data.get("positions", {})
            total_value = portfolio_data.get("total_value", 0)
            
            if total_value == 0:
                return portfolio_compliance
            
            # Check position limits
            for symbol, position in positions.items():
                position_value = position.get("position_value", 0)
                position_ratio = position_value / total_value
                
                portfolio_compliance["position_limits"][symbol] = {
                    "value": position_value,
                    "ratio": position_ratio,
                    "compliant": position_ratio <= self.compliance_policies["position_limits"]["max_single_position"]
                }
            
            # Check diversification
            num_positions = len(positions)
            portfolio_compliance["diversification"] = {
                "num_positions": num_positions,
                "compliant": num_positions >= self.compliance_policies["position_limits"]["min_diversification"]
            }
            
            # Check leverage (simplified)
            leverage = portfolio_data.get("leverage", 1.0)
            portfolio_compliance["leverage"] = {
                "current_leverage": leverage,
                "compliant": leverage <= self.compliance_policies["position_limits"]["max_leverage"]
            }
            
            # Check for restricted securities
            restricted_securities = []
            for symbol in positions.keys():
                if self._is_restricted_security(symbol):
                    restricted_securities.append(symbol)
            
            portfolio_compliance["restricted_securities"] = {
                "restricted_positions": restricted_securities,
                "compliant": len(restricted_securities) == 0
            }
            
            return portfolio_compliance
            
        except Exception as e:
            logger.error(f"Error checking portfolio compliance: {str(e)}")
            return {"error": str(e)}
    
    async def _check_trade_compliance(self, trade_data: Dict[str, Any], portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check trade-level compliance."""
        try:
            trade_compliance = {
                "pre_trade_checks": {},
                "post_trade_impact": {},
                "trading_restrictions": {}
            }
            
            symbol = trade_data.get("symbol", "")
            action = trade_data.get("action", "")
            quantity = trade_data.get("quantity", 0)
            current_price = trade_data.get("price", 0)
            trade_value = quantity * current_price
            
            # Pre-trade checks
            pre_trade_checks = {}
            
            # Check if security is restricted
            pre_trade_checks["restricted_security"] = {
                "restricted": self._is_restricted_security(symbol),
                "compliant": not self._is_restricted_security(symbol)
            }
            
            # Check trading hours
            current_time = datetime.now().time()
            trading_start = datetime.strptime(self.compliance_policies["trading_restrictions"]["trading_hours"]["start"], "%H:%M").time()
            trading_end = datetime.strptime(self.compliance_policies["trading_restrictions"]["trading_hours"]["end"], "%H:%M").time()
            
            pre_trade_checks["trading_hours"] = {
                "current_time": current_time.strftime("%H:%M"),
                "trading_hours": f"{trading_start.strftime('%H:%M')}-{trading_end.strftime('%H:%M')}",
                "compliant": trading_start <= current_time <= trading_end
            }
            
            # Check blackout periods
            blackout_periods = self._check_blackout_periods(symbol)
            pre_trade_checks["blackout_periods"] = {
                "in_blackout": len(blackout_periods) > 0,
                "blackout_reasons": blackout_periods,
                "compliant": len(blackout_periods) == 0
            }
            
            trade_compliance["pre_trade_checks"] = pre_trade_checks
            
            # Post-trade impact analysis
            if action == "buy":
                # Check if trade would exceed position limits
                positions = portfolio_data.get("positions", {})
                current_position_value = positions.get(symbol, {}).get("position_value", 0)
                new_position_value = current_position_value + trade_value
                total_value = portfolio_data.get("total_value", 0)
                
                position_ratio = new_position_value / total_value if total_value > 0 else 0
                
                trade_compliance["post_trade_impact"] = {
                    "new_position_value": new_position_value,
                    "new_position_ratio": position_ratio,
                    "would_exceed_limit": position_ratio > self.compliance_policies["position_limits"]["max_single_position"]
                }
            
            # Trading restrictions summary
            all_checks_passed = all(
                check.get("compliant", False) 
                for check in pre_trade_checks.values()
            )
            
            trade_compliance["trading_restrictions"] = {
                "overall_compliant": all_checks_passed,
                "restrictions_applied": not all_checks_passed
            }
            
            return trade_compliance
            
        except Exception as e:
            logger.error(f"Error checking trade compliance: {str(e)}")
            return {"error": str(e)}
    
    def _is_restricted_security(self, symbol: str) -> bool:
        """Check if a security is restricted."""
        try:
            # Simplified restricted security check
            # In a real implementation, this would check against a comprehensive database
            
            restricted_patterns = [
                "PENNY_", "CRYPTO_UTILITY_", "OTC_", "PINK_"
            ]
            
            for pattern in restricted_patterns:
                if pattern in symbol.upper():
                    return True
            
            # Check specific restricted symbols
            restricted_symbols = ["PENNY_STOCKS", "CRYPTO_UTILITY_TOKENS"]
            return symbol.upper() in restricted_symbols
            
        except Exception as e:
            logger.error(f"Error checking restricted security: {str(e)}")
            return False
    
    def _check_blackout_periods(self, symbol: str) -> List[str]:
        """Check if symbol is in any blackout periods."""
        try:
            blackout_reasons = []
            
            # Simplified blackout period check
            # In a real implementation, this would check earnings calendars, insider trading windows, etc.
            
            # Example: Check if it's earnings season for major stocks
            earnings_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
            if symbol in earnings_stocks:
                # Simplified: assume earnings are in first week of each quarter
                current_month = datetime.now().month
                earnings_months = [1, 4, 7, 10]  # January, April, July, October
                
                if current_month in earnings_months:
                    blackout_reasons.append("EARNINGS_ANNOUNCEMENTS")
            
            return blackout_reasons
            
        except Exception as e:
            logger.error(f"Error checking blackout periods: {str(e)}")
            return []
    
    async def _identify_policy_violations(self, compliance_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify policy violations from compliance results."""
        try:
            violations = []
            
            # Check portfolio compliance violations
            portfolio_compliance = compliance_results.get("portfolio_compliance", {})
            
            # Position limit violations
            position_limits = portfolio_compliance.get("position_limits", {})
            for symbol, limit_check in position_limits.items():
                if not limit_check.get("compliant", True):
                    violations.append({
                        "type": "position_limit_violation",
                        "severity": "high",
                        "symbol": symbol,
                        "current_ratio": limit_check.get("ratio", 0),
                        "limit": self.compliance_policies["position_limits"]["max_single_position"],
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Diversification violations
            diversification = portfolio_compliance.get("diversification", {})
            if not diversification.get("compliant", True):
                violations.append({
                    "type": "diversification_violation",
                    "severity": "medium",
                    "current_positions": diversification.get("num_positions", 0),
                    "required_positions": self.compliance_policies["position_limits"]["min_diversification"],
                    "timestamp": datetime.now().isoformat()
                })
            
            # Leverage violations
            leverage = portfolio_compliance.get("leverage", {})
            if not leverage.get("compliant", True):
                violations.append({
                    "type": "leverage_violation",
                    "severity": "high",
                    "current_leverage": leverage.get("current_leverage", 0),
                    "limit": self.compliance_policies["position_limits"]["max_leverage"],
                    "timestamp": datetime.now().isoformat()
                })
            
            # Restricted securities violations
            restricted_securities = portfolio_compliance.get("restricted_securities", {})
            if not restricted_securities.get("compliant", True):
                violations.append({
                    "type": "restricted_security_violation",
                    "severity": "high",
                    "restricted_symbols": restricted_securities.get("restricted_positions", []),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Check trade compliance violations
            trade_compliance = compliance_results.get("trade_compliance", {})
            pre_trade_checks = trade_compliance.get("pre_trade_checks", {})
            
            for check_type, check_result in pre_trade_checks.items():
                if not check_result.get("compliant", True):
                    violations.append({
                        "type": f"trade_{check_type}_violation",
                        "severity": "medium",
                        "check_type": check_type,
                        "details": check_result,
                        "timestamp": datetime.now().isoformat()
                    })
            
            return violations
            
        except Exception as e:
            logger.error(f"Error identifying policy violations: {str(e)}")
            return []
    
    async def _generate_compliance_recommendations(self, compliance_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate compliance recommendations."""
        try:
            recommendations = []
            violations = compliance_results.get("policy_violations", [])
            
            for violation in violations:
                violation_type = violation.get("type", "")
                
                if violation_type == "position_limit_violation":
                    recommendations.append({
                        "type": "reduce_position",
                        "priority": "high",
                        "symbol": violation.get("symbol"),
                        "action": f"Reduce position in {violation.get('symbol')} to comply with position limits",
                        "current_ratio": violation.get("current_ratio", 0),
                        "target_ratio": violation.get("limit", 0)
                    })
                
                elif violation_type == "diversification_violation":
                    recommendations.append({
                        "type": "increase_diversification",
                        "priority": "medium",
                        "action": "Add more positions to meet diversification requirements",
                        "current_positions": violation.get("current_positions", 0),
                        "required_positions": violation.get("required_positions", 0)
                    })
                
                elif violation_type == "leverage_violation":
                    recommendations.append({
                        "type": "reduce_leverage",
                        "priority": "high",
                        "action": "Reduce portfolio leverage to comply with limits",
                        "current_leverage": violation.get("current_leverage", 0),
                        "target_leverage": violation.get("limit", 0)
                    })
                
                elif violation_type == "restricted_security_violation":
                    recommendations.append({
                        "type": "remove_restricted_securities",
                        "priority": "high",
                        "action": "Remove positions in restricted securities",
                        "restricted_symbols": violation.get("restricted_symbols", [])
                    })
                
                elif violation_type.startswith("trade_"):
                    recommendations.append({
                        "type": "delay_trade",
                        "priority": "medium",
                        "action": "Delay trade until compliance issues are resolved",
                        "violation_type": violation_type
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating compliance recommendations: {str(e)}")
            return []
    
    async def _log_compliance_check(self, compliance_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Log compliance check for audit trail."""
        try:
            audit_entries = []
            
            # Log overall compliance status
            audit_entries.append({
                "timestamp": datetime.now().isoformat(),
                "event_type": "compliance_check",
                "overall_compliance": compliance_results.get("overall_compliance", False),
                "violations_count": len(compliance_results.get("policy_violations", [])),
                "recommendations_count": len(compliance_results.get("recommendations", []))
            })
            
            # Log individual violations
            for violation in compliance_results.get("policy_violations", []):
                audit_entries.append({
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "policy_violation",
                    "violation_type": violation.get("type"),
                    "severity": violation.get("severity"),
                    "details": violation
                })
            
            # Add to audit trail
            self.audit_trail.extend(audit_entries)
            
            return audit_entries
            
        except Exception as e:
            logger.error(f"Error logging compliance check: {str(e)}")
            return []
    
    async def _continuous_compliance_monitoring(self):
        """Continuous compliance monitoring task."""
        while self.is_running:
            try:
                # This would typically monitor real-time portfolio and trading data
                # For demo purposes, we'll simulate periodic monitoring
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in continuous compliance monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def _policy_enforcement_monitoring(self):
        """Policy enforcement monitoring task."""
        while self.is_running:
            try:
                # Check for policy violations and enforce policies
                enforcement_actions = await self.policy_enforcer.check_and_enforce()
                
                if enforcement_actions:
                    logger.info(f"Policy enforcement actions taken: {len(enforcement_actions)}")
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in policy enforcement monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def _audit_trail_monitoring(self):
        """Audit trail monitoring task."""
        while self.is_running:
            try:
                # Monitor audit trail and ensure proper logging
                audit_status = await self.audit_logger.check_audit_trail()
                
                if not audit_status.get("healthy", True):
                    logger.warning("Audit trail health issues detected")
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in audit trail monitoring: {str(e)}")
                await asyncio.sleep(300)
    
    async def _regulatory_reporting(self):
        """Regulatory reporting task."""
        while self.is_running:
            try:
                # Generate regulatory reports
                daily_report = await self.generate_daily_compliance_report()
                self.compliance_reports.append(daily_report)
                
                logger.info("Daily compliance report generated")
                
                await asyncio.sleep(86400)  # Daily report
                
            except Exception as e:
                logger.error(f"Error in regulatory reporting: {str(e)}")
                await asyncio.sleep(3600)
    
    async def generate_daily_compliance_report(self) -> Dict[str, Any]:
        """Generate daily compliance report."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "report_type": "daily_compliance",
                "overall_compliance_status": len(self.violations) == 0,
                "violations_summary": {
                    "total_violations": len(self.violations),
                    "high_severity": len([v for v in self.violations if v.get("severity") == "high"]),
                    "medium_severity": len([v for v in self.violations if v.get("severity") == "medium"]),
                    "low_severity": len([v for v in self.violations if v.get("severity") == "low"])
                },
                "audit_trail_summary": {
                    "total_entries": len(self.audit_trail),
                    "entries_today": len([e for e in self.audit_trail 
                                        if datetime.fromisoformat(e["timestamp"]).date() == datetime.now().date()])
                },
                "policy_compliance": {
                    "position_limits": "compliant",
                    "diversification": "compliant",
                    "leverage": "compliant",
                    "restricted_securities": "compliant"
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating daily compliance report: {str(e)}")
            return {"error": str(e)}
    
    async def get_compliance_summary(self) -> Dict[str, Any]:
        """Get current compliance summary."""
        try:
            return {
                "overall_compliance": len(self.violations) == 0,
                "total_violations": len(self.violations),
                "recent_violations": self.violations[-10:] if self.violations else [],
                "audit_trail_entries": len(self.audit_trail),
                "compliance_reports": len(self.compliance_reports),
                "last_report": self.compliance_reports[-1] if self.compliance_reports else None
            }
            
        except Exception as e:
            logger.error(f"Error getting compliance summary: {str(e)}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown the Compliance Officer AI Employee."""
        try:
            logger.info(f"Shutting down Compliance Officer AI Employee {self.employee_id}")
            
            # Stop background tasks
            self.is_running = False
            
            # Generate final compliance report
            final_report = await self.generate_daily_compliance_report()
            logger.info(f"Final compliance report - Overall compliance: {final_report.get('overall_compliance_status', False)}")
            
            await super().shutdown()
            
        except Exception as e:
            logger.error(f"Error during Compliance Officer shutdown: {str(e)}") 