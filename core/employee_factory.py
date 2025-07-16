"""
AI Employee Factory for creating and managing specialized AI Employees.
Handles the creation, training, and lifecycle management of AI agents.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
from dataclasses import dataclass
import uuid

from config.settings import settings
from employees.base_employee import BaseAIEmployee
from employees.research_analyst import ResearchAnalyst
from employees.trader import Trader
from employees.risk_manager import RiskManager
from employees.compliance_officer import ComplianceOfficer
from employees.data_specialist import DataSpecialist
from monitoring.metrics import MetricsCollector
from data.data_manager import DataManager

logger = logging.getLogger(__name__)

@dataclass
class EmployeeSpec:
    """Specification for creating an AI Employee."""
    role: str
    specialization: str
    model_architecture: str
    training_data_sources: List[str]
    evaluation_metrics: List[str]
    optimization_focus: List[str]
    learning_rate: float
    batch_size: int
    epochs: int
    created_at: datetime
    employee_id: str

class AIEmployeeFactory:
    """
    Factory for creating and managing AI Employees.
    
    Responsibilities:
    - Create specialized AI Employees based on role requirements
    - Manage training and deployment of AI Employees
    - Handle lifecycle management and optimization
    - Coordinate with data and monitoring systems
    """
    
    def __init__(self):
        """Initialize the AI Employee Factory."""
        self.employee_classes: Dict[str, Type[BaseAIEmployee]] = {
            "research_analyst": ResearchAnalyst,
            "trader": Trader,
            "risk_manager": RiskManager,
            "compliance_officer": ComplianceOfficer,
            "data_specialist": DataSpecialist
        }
        
        self.active_employees: Dict[str, BaseAIEmployee] = {}
        self.employee_specs: Dict[str, EmployeeSpec] = {}
        self.metrics_collector = MetricsCollector()
        self.data_manager = DataManager()
        
        logger.info("AI Employee Factory initialized")
    
    async def create_ai_employee(
        self, 
        role: str, 
        specialization: str, 
        context: Any = None
    ) -> str:
        """
        Create a new AI Employee with the specified role and specialization.
        
        Args:
            role: The role of the AI Employee (research_analyst, trader, etc.)
            specialization: Specific focus area or market
            context: Conversation context for additional parameters
            
        Returns:
            Employee ID of the created AI Employee
        """
        try:
            # Validate role
            if role not in self.employee_classes:
                raise ValueError(f"Invalid role: {role}. Available roles: {list(self.employee_classes.keys())}")
            
            # Generate unique employee ID
            employee_id = self._generate_employee_id(role)
            
            # Get role configuration
            role_config = settings.ai_employee_roles.get(role, {})
            
            # Create employee specification
            spec = EmployeeSpec(
                role=role,
                specialization=specialization,
                model_architecture=role_config.get("model_architecture", "default"),
                training_data_sources=role_config.get("training_data_sources", []),
                evaluation_metrics=role_config.get("evaluation_metrics", []),
                optimization_focus=role_config.get("optimization_focus", []),
                learning_rate=role_config.get("learning_rate", 0.001),
                batch_size=role_config.get("batch_size", 32),
                epochs=role_config.get("epochs", 100),
                created_at=datetime.now(),
                employee_id=employee_id
            )
            
            # Create AI Employee instance
            employee_class = self.employee_classes[role]
            employee = employee_class(
                employee_id=employee_id,
                spec=spec,
                context=context
            )
            
            # Store employee and spec
            self.active_employees[employee_id] = employee
            self.employee_specs[employee_id] = spec
            
            # Initialize employee
            await employee.initialize()
            
            # Start training in background
            asyncio.create_task(self._train_employee(employee_id))
            
            logger.info(f"Created AI Employee {employee_id} with role {role} and specialization {specialization}")
            
            return employee_id
            
        except Exception as e:
            logger.error(f"Error creating AI Employee: {str(e)}")
            raise
    
    async def _train_employee(self, employee_id: str):
        """Train an AI Employee in the background."""
        try:
            employee = self.active_employees.get(employee_id)
            if not employee:
                logger.error(f"Employee {employee_id} not found for training")
                return
            
            logger.info(f"Starting training for AI Employee {employee_id}")
            
            # Update status
            await self.metrics_collector.update_employee_status(employee_id, "training")
            
            # Prepare training data
            training_data = await self._prepare_training_data(employee_id)
            
            # Start training
            training_result = await employee.train(training_data)
            
            # Update status based on training result
            if training_result.get("success", False):
                await self.metrics_collector.update_employee_status(employee_id, "active")
                logger.info(f"Training completed successfully for AI Employee {employee_id}")
            else:
                await self.metrics_collector.update_employee_status(employee_id, "failed")
                logger.error(f"Training failed for AI Employee {employee_id}: {training_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Error during training for AI Employee {employee_id}: {str(e)}")
            await self.metrics_collector.update_employee_status(employee_id, "failed")
    
    async def _prepare_training_data(self, employee_id: str) -> Dict[str, Any]:
        """Prepare training data for an AI Employee."""
        try:
            spec = self.employee_specs.get(employee_id)
            if not spec:
                raise ValueError(f"Specification not found for employee {employee_id}")
            
            training_data = {
                "role": spec.role,
                "specialization": spec.specialization,
                "data_sources": spec.training_data_sources,
                "parameters": {
                    "learning_rate": spec.learning_rate,
                    "batch_size": spec.batch_size,
                    "epochs": spec.epochs
                }
            }
            
            # Get data from data manager
            for data_source in spec.training_data_sources:
                data = await self.data_manager.get_training_data(data_source, spec.specialization)
                training_data[data_source] = data
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error preparing training data for employee {employee_id}: {str(e)}")
            raise
    
    async def get_employee(self, employee_id: str) -> Optional[BaseAIEmployee]:
        """Get an AI Employee by ID."""
        return self.active_employees.get(employee_id)
    
    async def get_employee_status(self, employee_id: str) -> Dict[str, Any]:
        """Get the status of an AI Employee."""
        try:
            employee = self.active_employees.get(employee_id)
            if not employee:
                return {"status": "not_found", "error": "Employee not found"}
            
            spec = self.employee_specs.get(employee_id)
            if not spec:
                return {"status": "error", "error": "Specification not found"}
            
            # Get current status from metrics
            metrics = await self.metrics_collector.get_employee_metrics(employee_id)
            
            return {
                "employee_id": employee_id,
                "role": spec.role,
                "specialization": spec.specialization,
                "status": metrics.get("status", "unknown"),
                "created_at": spec.created_at.isoformat(),
                "metrics": metrics,
                "spec": asdict(spec)
            }
            
        except Exception as e:
            logger.error(f"Error getting employee status for {employee_id}: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def get_all_employees(self) -> List[Dict[str, Any]]:
        """Get status of all AI Employees."""
        try:
            employees = []
            for employee_id in self.active_employees.keys():
                status = await self.get_employee_status(employee_id)
                employees.append(status)
            return employees
            
        except Exception as e:
            logger.error(f"Error getting all employees: {str(e)}")
            return []
    
    async def optimize_employee(self, employee_id: str, optimization_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize an AI Employee."""
        try:
            employee = self.active_employees.get(employee_id)
            if not employee:
                return {"success": False, "error": "Employee not found"}
            
            logger.info(f"Starting optimization for AI Employee {employee_id}")
            
            # Update status
            await self.metrics_collector.update_employee_status(employee_id, "optimizing")
            
            # Perform optimization
            optimization_result = await employee.optimize(optimization_params)
            
            # Update status
            if optimization_result.get("success", False):
                await self.metrics_collector.update_employee_status(employee_id, "active")
                logger.info(f"Optimization completed for AI Employee {employee_id}")
            else:
                await self.metrics_collector.update_employee_status(employee_id, "optimization_failed")
                logger.error(f"Optimization failed for AI Employee {employee_id}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error optimizing employee {employee_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def stop_ai_employee(self, employee_id: str) -> bool:
        """Stop and cleanup an AI Employee."""
        try:
            employee = self.active_employees.get(employee_id)
            if not employee:
                logger.warning(f"Employee {employee_id} not found for stopping")
                return False
            
            # Stop the employee
            await employee.stop()
            
            # Remove from active employees
            del self.active_employees[employee_id]
            if employee_id in self.employee_specs:
                del self.employee_specs[employee_id]
            
            logger.info(f"Stopped AI Employee {employee_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping AI Employee {employee_id}: {str(e)}")
            return False
    
    async def restart_ai_employee(self, employee_id: str) -> bool:
        """Restart an AI Employee."""
        try:
            employee = self.active_employees.get(employee_id)
            if not employee:
                logger.warning(f"Employee {employee_id} not found for restarting")
                return False
            
            # Restart the employee
            await employee.restart()
            
            logger.info(f"Restarted AI Employee {employee_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error restarting AI Employee {employee_id}: {str(e)}")
            return False
    
    def _generate_employee_id(self, role: str) -> str:
        """Generate a unique employee ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = str(uuid.uuid4())[:8]
        return f"{role}_{timestamp}_{random_suffix}"
    
    async def get_employee_capabilities(self, role: str) -> Dict[str, Any]:
        """Get capabilities for a specific role."""
        try:
            role_config = settings.ai_employee_roles.get(role, {})
            
            return {
                "role": role,
                "model_architecture": role_config.get("model_architecture", "N/A"),
                "training_data_sources": role_config.get("training_data_sources", []),
                "evaluation_metrics": role_config.get("evaluation_metrics", []),
                "optimization_focus": role_config.get("optimization_focus", []),
                "training_parameters": {
                    "learning_rate": role_config.get("learning_rate", "N/A"),
                    "batch_size": role_config.get("batch_size", "N/A"),
                    "epochs": role_config.get("epochs", "N/A")
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting capabilities for role {role}: {str(e)}")
            return {"error": str(e)}
    
    async def get_all_roles(self) -> List[str]:
        """Get all available AI Employee roles."""
        return list(self.employee_classes.keys())
    
    async def validate_role_specialization(self, role: str, specialization: str) -> Dict[str, Any]:
        """Validate if a specialization is appropriate for a role."""
        try:
            # Get role capabilities
            capabilities = await self.get_employee_capabilities(role)
            
            # Basic validation
            validation_result = {
                "valid": True,
                "role": role,
                "specialization": specialization,
                "warnings": [],
                "recommendations": []
            }
            
            # Check if role exists
            if role not in self.employee_classes:
                validation_result["valid"] = False
                validation_result["warnings"].append(f"Role '{role}' is not supported")
                return validation_result
            
            # Role-specific validation
            if role == "research_analyst":
                if not any(keyword in specialization.lower() for keyword in ["market", "sector", "asset", "crypto", "forex", "equity"]):
                    validation_result["warnings"].append("Research Analyst typically focuses on specific markets or asset classes")
                    validation_result["recommendations"].append("Consider specifying a market (e.g., 'cryptocurrency markets', 'tech sector')")
            
            elif role == "trader":
                if not any(keyword in specialization.lower() for keyword in ["trading", "execution", "strategy", "algorithm", "market"]):
                    validation_result["warnings"].append("Trader typically focuses on trading strategies or execution methods")
                    validation_result["recommendations"].append("Consider specifying a trading approach (e.g., 'high-frequency trading', 'algorithmic trading')")
            
            elif role == "risk_manager":
                if not any(keyword in specialization.lower() for keyword in ["risk", "portfolio", "var", "stress", "compliance"]):
                    validation_result["warnings"].append("Risk Manager typically focuses on risk assessment or portfolio management")
                    validation_result["recommendations"].append("Consider specifying risk focus (e.g., 'portfolio risk', 'market risk', 'credit risk')")
            
            elif role == "compliance_officer":
                if not any(keyword in specialization.lower() for keyword in ["compliance", "regulatory", "audit", "legal", "policy"]):
                    validation_result["warnings"].append("Compliance Officer typically focuses on regulatory compliance")
                    validation_result["recommendations"].append("Consider specifying compliance area (e.g., 'SEC compliance', 'regulatory reporting')")
            
            elif role == "data_specialist":
                if not any(keyword in specialization.lower() for keyword in ["data", "cleaning", "processing", "quality", "pipeline"]):
                    validation_result["warnings"].append("Data Specialist typically focuses on data management")
                    validation_result["recommendations"].append("Consider specifying data focus (e.g., 'data cleaning', 'data pipeline optimization')")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating role specialization: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "role": role,
                "specialization": specialization
            }
    
    async def get_employee_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of all AI Employee performance."""
        try:
            summary = {
                "total_employees": len(self.active_employees),
                "by_role": {},
                "by_status": {},
                "performance_metrics": {}
            }
            
            # Group by role
            for employee_id, employee in self.active_employees.items():
                spec = self.employee_specs.get(employee_id)
                if spec:
                    role = spec.role
                    if role not in summary["by_role"]:
                        summary["by_role"][role] = 0
                    summary["by_role"][role] += 1
            
            # Get status distribution
            for employee_id in self.active_employees.keys():
                status = await self.metrics_collector.get_employee_status(employee_id)
                if status not in summary["by_status"]:
                    summary["by_status"][status] = 0
                summary["by_status"][status] += 1
            
            # Get performance metrics
            all_metrics = await self.metrics_collector.get_all_metrics()
            summary["performance_metrics"] = all_metrics
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {"error": str(e)}
    
    async def cleanup_inactive_employees(self, max_inactive_hours: int = 24):
        """Clean up inactive employees."""
        try:
            current_time = datetime.now()
            inactive_employees = []
            
            for employee_id, spec in self.employee_specs.items():
                # Check if employee has been inactive for too long
                time_diff = current_time - spec.created_at
                if time_diff.total_seconds() > max_inactive_hours * 3600:
                    inactive_employees.append(employee_id)
            
            # Stop inactive employees
            for employee_id in inactive_employees:
                await self.stop_ai_employee(employee_id)
            
            if inactive_employees:
                logger.info(f"Cleaned up {len(inactive_employees)} inactive employees")
                
        except Exception as e:
            logger.error(f"Error cleaning up inactive employees: {str(e)}")
    
    async def shutdown(self):
        """Shutdown the factory and all employees."""
        logger.info("Shutting down AI Employee Factory...")
        
        # Stop all employees
        for employee_id in list(self.active_employees.keys()):
            await self.stop_ai_employee(employee_id)
        
        logger.info("AI Employee Factory shutdown complete") 