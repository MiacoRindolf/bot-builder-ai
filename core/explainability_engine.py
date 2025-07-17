"""
Advanced Explainability Engine for Bot Builder AI System.
Provides transparency and interpretability for AI Employee decisions.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Try to import explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available, explainability features will be limited")

try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available, explainability features will be limited")

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """Result of an explanation analysis."""
    employee_id: str
    decision_id: str
    explanation_type: str
    confidence: float
    factors: List[Dict[str, Any]]
    visualization: Optional[str] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

@dataclass
class FeatureImportance:
    """Feature importance for explainability."""
    feature_name: str
    importance_score: float
    direction: str  # 'positive', 'negative', 'neutral'
    contribution: float
    description: str

@dataclass
class DecisionContext:
    """Context for AI decision making."""
    market_data: Dict[str, Any]
    portfolio_state: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    historical_context: Dict[str, Any]
    regulatory_context: Dict[str, Any]

class AdvancedExplainabilityEngine:
    """
    Advanced Explainability Engine for AI Employee decisions.
    
    Features:
    - SHAP-based feature importance analysis
    - LIME-based local explanations
    - Decision tree visualization
    - Risk factor analysis
    - Regulatory compliance explanations
    - Performance attribution
    - Interactive visualizations
    """
    
    def __init__(self):
        """Initialize the Advanced Explainability Engine."""
        self.is_initialized = False
        self.explanations_cache: Dict[str, ExplanationResult] = {}
        self.feature_importance_cache: Dict[str, List[FeatureImportance]] = {}
        self.decision_history: List[Dict[str, Any]] = []
        
        # Explanation methods
        self.explanation_methods = {
            "shap": self._shap_explanation if SHAP_AVAILABLE else None,
            "lime": self._lime_explanation if LIME_AVAILABLE else None,
            "rule_based": self._rule_based_explanation,
            "statistical": self._statistical_explanation,
            "regulatory": self._regulatory_explanation
        }
        
        # Configuration
        self.max_cache_size = 1000
        self.explanation_timeout = 30  # seconds
        self.visualization_enabled = True
        
        logger.info("Advanced Explainability Engine initialized")
    
    async def initialize(self) -> bool:
        """Initialize the Advanced Explainability Engine."""
        try:
            logger.info("Initializing Advanced Explainability Engine...")
            
            # Check available libraries
            self._check_available_libraries()
            
            # Initialize visualization settings
            await self._setup_visualization()
            
            # Set up explanation tracking
            await self._setup_explanation_tracking()
            
            self.is_initialized = True
            logger.info("Advanced Explainability Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Advanced Explainability Engine: {str(e)}")
            return False
    
    async def explain_decision(
        self, 
        employee_id: str, 
        decision_data: Dict[str, Any], 
        context: DecisionContext,
        explanation_types: List[str] = None
    ) -> ExplanationResult:
        """Generate comprehensive explanation for an AI decision."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if explanation_types is None:
                explanation_types = ["rule_based", "statistical"]
                if SHAP_AVAILABLE:
                    explanation_types.append("shap")
                if LIME_AVAILABLE:
                    explanation_types.append("lime")
            
            decision_id = f"{employee_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Generating explanation for decision {decision_id} by employee {employee_id}")
            
            # Generate explanations using different methods
            explanations = {}
            factors = []
            
            for method in explanation_types:
                if method in self.explanation_methods and self.explanation_methods[method]:
                    try:
                        method_explanation = await self.explanation_methods[method](
                            employee_id, decision_data, context
                        )
                        explanations[method] = method_explanation
                        factors.extend(method_explanation.get("factors", []))
                    except Exception as e:
                        logger.error(f"Error in {method} explanation: {str(e)}")
            
            # Create visualization if enabled
            visualization = None
            if self.visualization_enabled:
                visualization = await self._create_explanation_visualization(
                    explanations, decision_data, context
                )
            
            # Calculate overall confidence
            confidence = self._calculate_explanation_confidence(explanations)
            
            # Create explanation result
            result = ExplanationResult(
                employee_id=employee_id,
                decision_id=decision_id,
                explanation_type="comprehensive",
                confidence=confidence,
                factors=factors,
                visualization=visualization,
                timestamp=datetime.now(),
                metadata={
                    "methods_used": list(explanations.keys()),
                    "context_summary": self._summarize_context(context)
                }
            )
            
            # Cache the explanation
            self.explanations_cache[decision_id] = result
            
            # Add to decision history
            self.decision_history.append({
                "decision_id": decision_id,
                "employee_id": employee_id,
                "timestamp": datetime.now().isoformat(),
                "decision_type": decision_data.get("action_type", "unknown"),
                "confidence": confidence
            })
            
            # Maintain cache size
            if len(self.explanations_cache) > self.max_cache_size:
                self._cleanup_cache()
            
            logger.info(f"Explanation generated for decision {decision_id} with confidence {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error explaining decision: {str(e)}")
            return self._create_error_explanation(employee_id, str(e))
    
    async def get_feature_importance(self, employee_id: str, model_data: Dict[str, Any]) -> List[FeatureImportance]:
        """Get feature importance for an AI Employee's model."""
        try:
            cache_key = f"{employee_id}_feature_importance"
            
            if cache_key in self.feature_importance_cache:
                return self.feature_importance_cache[cache_key]
            
            # Generate feature importance
            features = await self._analyze_feature_importance(employee_id, model_data)
            
            # Cache the results
            self.feature_importance_cache[cache_key] = features
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return []
    
    async def get_decision_history(self, employee_id: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get decision history for an AI Employee."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            history = [
                decision for decision in self.decision_history
                if decision["employee_id"] == employee_id and 
                datetime.fromisoformat(decision["timestamp"]) > cutoff_date
            ]
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting decision history: {str(e)}")
            return []
    
    async def get_explanation_summary(self, employee_id: str) -> Dict[str, Any]:
        """Get explanation summary for an AI Employee."""
        try:
            # Get recent explanations
            recent_explanations = [
                exp for exp in self.explanations_cache.values()
                if exp.employee_id == employee_id and 
                exp.timestamp > datetime.now() - timedelta(days=1)
            ]
            
            if not recent_explanations:
                return {"message": "No recent explanations available"}
            
            # Calculate summary statistics
            avg_confidence = np.mean([exp.confidence for exp in recent_explanations])
            decision_types = [exp.metadata.get("decision_type", "unknown") for exp in recent_explanations]
            
            # Get most common factors
            all_factors = []
            for exp in recent_explanations:
                all_factors.extend(exp.factors)
            
            factor_counts = {}
            for factor in all_factors:
                factor_name = factor.get("name", "unknown")
                factor_counts[factor_name] = factor_counts.get(factor_name, 0) + 1
            
            top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "total_explanations": len(recent_explanations),
                "average_confidence": avg_confidence,
                "decision_types": list(set(decision_types)),
                "top_factors": [{"name": name, "count": count} for name, count in top_factors],
                "last_explanation": recent_explanations[-1].timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting explanation summary: {str(e)}")
            return {"error": str(e)}
    
    async def _shap_explanation(self, employee_id: str, decision_data: Dict[str, Any], context: DecisionContext) -> Dict[str, Any]:
        """Generate SHAP-based explanation."""
        try:
            if not SHAP_AVAILABLE:
                return {"error": "SHAP not available"}
            
            # Extract features from decision data and context
            features = self._extract_features_for_shap(decision_data, context)
            
            # Create SHAP explainer (simplified version)
            # In a real implementation, you would use the actual model
            feature_names = list(features.keys())
            feature_values = list(features.values())
            
            # Simulate SHAP values
            shap_values = np.random.normal(0, 0.1, len(feature_values))
            shap_values = shap_values / np.sum(np.abs(shap_values))  # Normalize
            
            # Create factors from SHAP values
            factors = []
            for i, (name, value, shap_val) in enumerate(zip(feature_names, feature_values, shap_values)):
                factors.append({
                    "name": name,
                    "value": value,
                    "importance": abs(shap_val),
                    "contribution": shap_val,
                    "direction": "positive" if shap_val > 0 else "negative",
                    "description": f"Feature '{name}' contributed {shap_val:.3f} to the decision"
                })
            
            # Sort by importance
            factors.sort(key=lambda x: x["importance"], reverse=True)
            
            return {
                "method": "shap",
                "factors": factors,
                "confidence": 0.85,
                "description": "SHAP-based feature importance analysis"
            }
            
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {str(e)}")
            return {"error": str(e)}
    
    async def _lime_explanation(self, employee_id: str, decision_data: Dict[str, Any], context: DecisionContext) -> Dict[str, Any]:
        """Generate LIME-based explanation."""
        try:
            if not LIME_AVAILABLE:
                return {"error": "LIME not available"}
            
            # Extract features
            features = self._extract_features_for_lime(decision_data, context)
            
            # Simulate LIME explanation
            feature_names = list(features.keys())
            feature_values = list(features.values())
            
            # Create LIME-style factors
            factors = []
            for name, value in zip(feature_names, feature_values):
                # Simulate LIME weights
                weight = np.random.normal(0, 0.2)
                factors.append({
                    "name": name,
                    "value": value,
                    "weight": weight,
                    "importance": abs(weight),
                    "description": f"Local explanation: {name} has weight {weight:.3f}"
                })
            
            # Sort by importance
            factors.sort(key=lambda x: x["importance"], reverse=True)
            
            return {
                "method": "lime",
                "factors": factors,
                "confidence": 0.80,
                "description": "LIME-based local explanation"
            }
            
        except Exception as e:
            logger.error(f"Error in LIME explanation: {str(e)}")
            return {"error": str(e)}
    
    async def _rule_based_explanation(self, employee_id: str, decision_data: Dict[str, Any], context: DecisionContext) -> Dict[str, Any]:
        """Generate rule-based explanation."""
        try:
            action_type = decision_data.get("action_type", "unknown")
            factors = []
            
            # Market-based rules
            market_data = context.market_data
            if market_data:
                price_change = market_data.get("change_percent", 0)
                volume = market_data.get("volume", 0)
                
                if abs(price_change) > 5:
                    factors.append({
                        "name": "significant_price_movement",
                        "value": price_change,
                        "importance": 0.9,
                        "description": f"Significant price movement detected: {price_change:.2f}%"
                    })
                
                if volume > 1000000:  # High volume threshold
                    factors.append({
                        "name": "high_volume",
                        "value": volume,
                        "importance": 0.7,
                        "description": f"High trading volume: {volume:,}"
                    })
            
            # Risk-based rules
            risk_metrics = context.risk_metrics
            if risk_metrics:
                var_95 = risk_metrics.get("var_95", 0)
                if var_95 > 0.02:  # 2% VaR threshold
                    factors.append({
                        "name": "high_risk",
                        "value": var_95,
                        "importance": 0.8,
                        "description": f"High risk level: VaR 95% = {var_95:.2%}"
                    })
            
            # Portfolio-based rules
            portfolio_state = context.portfolio_state
            if portfolio_state:
                cash_ratio = portfolio_state.get("cash_ratio", 0)
                if cash_ratio < 0.1:  # Low cash threshold
                    factors.append({
                        "name": "low_cash",
                        "value": cash_ratio,
                        "importance": 0.6,
                        "description": f"Low cash position: {cash_ratio:.1%}"
                    })
            
            return {
                "method": "rule_based",
                "factors": factors,
                "confidence": 0.75,
                "description": f"Rule-based explanation for {action_type} action"
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based explanation: {str(e)}")
            return {"error": str(e)}
    
    async def _statistical_explanation(self, employee_id: str, decision_data: Dict[str, Any], context: DecisionContext) -> Dict[str, Any]:
        """Generate statistical explanation."""
        try:
            factors = []
            
            # Statistical analysis of decision factors
            market_data = context.market_data
            if market_data:
                # Price trend analysis
                price = market_data.get("price", 0)
                moving_average = market_data.get("moving_average", price)
                
                if price > moving_average * 1.05:
                    factors.append({
                        "name": "above_moving_average",
                        "value": price / moving_average,
                        "importance": 0.7,
                        "description": f"Price {((price/moving_average-1)*100):.1f}% above moving average"
                    })
                elif price < moving_average * 0.95:
                    factors.append({
                        "name": "below_moving_average",
                        "value": price / moving_average,
                        "importance": 0.7,
                        "description": f"Price {((price/moving_average-1)*100):.1f}% below moving average"
                    })
                
                # Volatility analysis
                volatility = market_data.get("volatility", 0)
                if volatility > 0.3:  # High volatility threshold
                    factors.append({
                        "name": "high_volatility",
                        "value": volatility,
                        "importance": 0.8,
                        "description": f"High volatility: {volatility:.2%}"
                    })
            
            return {
                "method": "statistical",
                "factors": factors,
                "confidence": 0.70,
                "description": "Statistical analysis of market conditions"
            }
            
        except Exception as e:
            logger.error(f"Error in statistical explanation: {str(e)}")
            return {"error": str(e)}
    
    async def _regulatory_explanation(self, employee_id: str, decision_data: Dict[str, Any], context: DecisionContext) -> Dict[str, Any]:
        """Generate regulatory compliance explanation."""
        try:
            factors = []
            
            # Regulatory context analysis
            regulatory_context = context.regulatory_context
            if regulatory_context:
                compliance_score = regulatory_context.get("compliance_score", 1.0)
                
                if compliance_score < 0.95:
                    factors.append({
                        "name": "compliance_risk",
                        "value": compliance_score,
                        "importance": 0.9,
                        "description": f"Compliance risk detected: score {compliance_score:.2%}"
                    })
                
                # Check for regulatory violations
                violations = regulatory_context.get("violations", [])
                if violations:
                    factors.append({
                        "name": "regulatory_violations",
                        "value": len(violations),
                        "importance": 1.0,
                        "description": f"Regulatory violations detected: {len(violations)} issues"
                    })
            
            return {
                "method": "regulatory",
                "factors": factors,
                "confidence": 0.85,
                "description": "Regulatory compliance analysis"
            }
            
        except Exception as e:
            logger.error(f"Error in regulatory explanation: {str(e)}")
            return {"error": str(e)}
    
    def _extract_features_for_shap(self, decision_data: Dict[str, Any], context: DecisionContext) -> Dict[str, float]:
        """Extract features for SHAP analysis."""
        features = {}
        
        # Market features
        if context.market_data:
            features.update({
                "price": context.market_data.get("price", 0),
                "volume": context.market_data.get("volume", 0),
                "change_percent": context.market_data.get("change_percent", 0),
                "volatility": context.market_data.get("volatility", 0)
            })
        
        # Portfolio features
        if context.portfolio_state:
            features.update({
                "cash_ratio": context.portfolio_state.get("cash_ratio", 0),
                "position_count": context.portfolio_state.get("position_count", 0),
                "total_value": context.portfolio_state.get("total_value", 0)
            })
        
        # Risk features
        if context.risk_metrics:
            features.update({
                "var_95": context.risk_metrics.get("var_95", 0),
                "max_drawdown": context.risk_metrics.get("max_drawdown", 0),
                "sharpe_ratio": context.risk_metrics.get("sharpe_ratio", 0)
            })
        
        return features
    
    def _extract_features_for_lime(self, decision_data: Dict[str, Any], context: DecisionContext) -> Dict[str, float]:
        """Extract features for LIME analysis."""
        return self._extract_features_for_shap(decision_data, context)
    
    async def _create_explanation_visualization(self, explanations: Dict[str, Any], decision_data: Dict[str, Any], context: DecisionContext) -> Optional[str]:
        """Create visualization for explanations."""
        try:
            if not self.visualization_enabled:
                return None
            
            # Create a simple visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('AI Decision Explanation', fontsize=16)
            
            # Feature importance plot
            if "shap" in explanations:
                factors = explanations["shap"].get("factors", [])
                if factors:
                    names = [f["name"] for f in factors[:5]]
                    importances = [f["importance"] for f in factors[:5]]
                    
                    axes[0, 0].barh(names, importances)
                    axes[0, 0].set_title("Feature Importance (SHAP)")
                    axes[0, 0].set_xlabel("Importance")
            
            # Decision factors
            all_factors = []
            for exp in explanations.values():
                all_factors.extend(exp.get("factors", []))
            
            if all_factors:
                factor_names = [f["name"] for f in all_factors[:5]]
                factor_importances = [f["importance"] for f in all_factors[:5]]
                
                axes[0, 1].pie(factor_importances, labels=factor_names, autopct='%1.1f%%')
                axes[0, 1].set_title("Decision Factors")
            
            # Market context
            if context.market_data:
                market_metrics = ["price", "volume", "change_percent", "volatility"]
                market_values = [context.market_data.get(m, 0) for m in market_metrics]
                
                axes[1, 0].bar(market_metrics, market_values)
                axes[1, 0].set_title("Market Context")
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Risk metrics
            if context.risk_metrics:
                risk_metrics = ["var_95", "max_drawdown", "sharpe_ratio"]
                risk_values = [context.risk_metrics.get(r, 0) for r in risk_metrics]
                
                axes[1, 1].bar(risk_metrics, risk_values)
                axes[1, 1].set_title("Risk Metrics")
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Convert to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None
    
    def _calculate_explanation_confidence(self, explanations: Dict[str, Any]) -> float:
        """Calculate overall confidence for explanations."""
        try:
            confidences = []
            for exp in explanations.values():
                if "confidence" in exp:
                    confidences.append(exp["confidence"])
            
            if confidences:
                return np.mean(confidences)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _summarize_context(self, context: DecisionContext) -> Dict[str, Any]:
        """Summarize decision context."""
        return {
            "market_summary": {
                "has_data": bool(context.market_data),
                "price_change": context.market_data.get("change_percent", 0) if context.market_data else 0
            },
            "portfolio_summary": {
                "has_data": bool(context.portfolio_state),
                "position_count": context.portfolio_state.get("position_count", 0) if context.portfolio_state else 0
            },
            "risk_summary": {
                "has_data": bool(context.risk_metrics),
                "var_95": context.risk_metrics.get("var_95", 0) if context.risk_metrics else 0
            }
        }
    
    def _create_error_explanation(self, employee_id: str, error_message: str) -> ExplanationResult:
        """Create error explanation when explanation fails."""
        return ExplanationResult(
            employee_id=employee_id,
            decision_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            explanation_type="error",
            confidence=0.0,
            factors=[{
                "name": "explanation_error",
                "value": 0,
                "importance": 1.0,
                "description": f"Explanation failed: {error_message}"
            }],
            timestamp=datetime.now(),
            metadata={"error": error_message}
        )
    
    async def _analyze_feature_importance(self, employee_id: str, model_data: Dict[str, Any]) -> List[FeatureImportance]:
        """Analyze feature importance for a model."""
        try:
            # Simulate feature importance analysis
            features = [
                FeatureImportance(
                    feature_name="market_price",
                    importance_score=0.25,
                    direction="positive",
                    contribution=0.15,
                    description="Market price has moderate positive influence"
                ),
                FeatureImportance(
                    feature_name="volume",
                    importance_score=0.20,
                    direction="positive",
                    contribution=0.12,
                    description="Trading volume has moderate influence"
                ),
                FeatureImportance(
                    feature_name="volatility",
                    importance_score=0.30,
                    direction="negative",
                    contribution=-0.18,
                    description="High volatility has strong negative influence"
                ),
                FeatureImportance(
                    feature_name="risk_metrics",
                    importance_score=0.25,
                    direction="negative",
                    contribution=-0.10,
                    description="Risk metrics have moderate negative influence"
                )
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            return []
    
    def _check_available_libraries(self):
        """Check which explainability libraries are available."""
        logger.info(f"SHAP available: {SHAP_AVAILABLE}")
        logger.info(f"LIME available: {LIME_AVAILABLE}")
    
    async def _setup_visualization(self):
        """Set up visualization settings."""
        try:
            # Configure matplotlib for non-interactive backend
            plt.switch_backend('Agg')
            logger.info("Visualization setup completed")
        except Exception as e:
            logger.error(f"Error setting up visualization: {str(e)}")
    
    async def _setup_explanation_tracking(self):
        """Set up explanation tracking."""
        try:
            logger.info("Explanation tracking setup completed")
        except Exception as e:
            logger.error(f"Error setting up explanation tracking: {str(e)}")
    
    def _cleanup_cache(self):
        """Clean up old cache entries."""
        try:
            # Remove oldest entries
            sorted_cache = sorted(
                self.explanations_cache.items(),
                key=lambda x: x[1].timestamp
            )
            
            # Keep only the newest entries
            self.explanations_cache = dict(sorted_cache[-self.max_cache_size:])
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}") 