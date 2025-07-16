"""
Base AI Employee class that defines the common interface and functionality
for all specialized AI Employees in the Bot Builder AI system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import json

from config.settings import settings
from monitoring.metrics import MetricsCollector
from data.data_manager import DataManager

logger = logging.getLogger(__name__)

@dataclass
class EmployeeMetrics:
    """Metrics for tracking AI Employee performance."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    execution_speed: float = 0.0
    success_rate: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class EmployeeStatus:
    """Status information for an AI Employee."""
    status: str = "initializing"  # initializing, training, active, paused, failed, optimizing
    is_online: bool = False
    last_activity: datetime = None
    error_count: int = 0
    success_count: int = 0
    total_operations: int = 0
    
    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.now()

class BaseAIEmployee(ABC):
    """
    Base class for all AI Employees in the Bot Builder AI system.
    
    This abstract class defines the common interface and functionality
    that all specialized AI Employees must implement.
    """
    
    def __init__(self, employee_id: str, spec: Any, context: Any = None):
        """
        Initialize the AI Employee.
        
        Args:
            employee_id: Unique identifier for this employee
            spec: Employee specification containing role and configuration
            context: Additional context for initialization
        """
        self.employee_id = employee_id
        self.spec = spec
        self.context = context
        
        # Core components
        self.metrics_collector = MetricsCollector()
        self.data_manager = DataManager()
        
        # State management
        self.status = EmployeeStatus()
        self.metrics = EmployeeMetrics()
        self.model = None
        self.is_initialized = False
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.operation_log: List[Dict[str, Any]] = []
        
        # Configuration
        self.config = self._load_config()
        
        logger.info(f"Initialized {self.__class__.__name__} with ID {employee_id}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the AI Employee with its specific capabilities.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the AI Employee with provided data.
        
        Args:
            training_data: Data and parameters for training
            
        Returns:
            Training result with success status and metrics
        """
        pass
    
    @abstractmethod
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task specific to this AI Employee's role.
        
        Args:
            task_data: Task parameters and data
            
        Returns:
            Task execution result
        """
        pass
    
    @abstractmethod
    async def optimize(self, optimization_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize the AI Employee based on performance and parameters.
        
        Args:
            optimization_params: Optimization parameters and focus areas
            
        Returns:
            Optimization result with improvements and metrics
        """
        pass
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of the AI Employee."""
        return {
            "employee_id": self.employee_id,
            "role": self.spec.role,
            "specialization": self.spec.specialization,
            "status": asdict(self.status),
            "metrics": asdict(self.metrics),
            "is_initialized": self.is_initialized,
            "config": self.config,
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_performance_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance history of the AI Employee."""
        return self.performance_history[-limit:] if self.performance_history else []
    
    async def get_operation_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent operation log of the AI Employee."""
        return self.operation_log[-limit:] if self.operation_log else []
    
    async def pause(self) -> bool:
        """Pause the AI Employee."""
        try:
            self.status.status = "paused"
            self.status.is_online = False
            await self.metrics_collector.update_employee_status(self.employee_id, "paused")
            logger.info(f"Paused AI Employee {self.employee_id}")
            return True
        except Exception as e:
            logger.error(f"Error pausing AI Employee {self.employee_id}: {str(e)}")
            return False
    
    async def resume(self) -> bool:
        """Resume the AI Employee."""
        try:
            self.status.status = "active"
            self.status.is_online = True
            await self.metrics_collector.update_employee_status(self.employee_id, "active")
            logger.info(f"Resumed AI Employee {self.employee_id}")
            return True
        except Exception as e:
            logger.error(f"Error resuming AI Employee {self.employee_id}: {str(e)}")
            return False
    
    async def restart(self) -> bool:
        """Restart the AI Employee."""
        try:
            logger.info(f"Restarting AI Employee {self.employee_id}")
            
            # Stop current operations
            await self.stop()
            
            # Reinitialize
            success = await self.initialize()
            if success:
                self.status.status = "active"
                self.status.is_online = True
                await self.metrics_collector.update_employee_status(self.employee_id, "active")
                logger.info(f"Successfully restarted AI Employee {self.employee_id}")
                return True
            else:
                logger.error(f"Failed to reinitialize AI Employee {self.employee_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error restarting AI Employee {self.employee_id}: {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """Stop the AI Employee."""
        try:
            self.status.status = "stopped"
            self.status.is_online = False
            
            # Cleanup resources
            await self._cleanup()
            
            await self.metrics_collector.update_employee_status(self.employee_id, "stopped")
            logger.info(f"Stopped AI Employee {self.employee_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping AI Employee {self.employee_id}: {str(e)}")
            return False
    
    async def update_metrics(self, new_metrics: Dict[str, Any]):
        """Update the AI Employee's metrics."""
        try:
            # Update metrics
            for key, value in new_metrics.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)
            
            self.metrics.last_updated = datetime.now()
            
            # Add to performance history
            self.performance_history.append({
                "timestamp": datetime.now().isoformat(),
                "metrics": asdict(self.metrics),
                "status": self.status.status
            })
            
            # Update metrics collector
            await self.metrics_collector.update_employee_metrics(self.employee_id, asdict(self.metrics))
            
        except Exception as e:
            logger.error(f"Error updating metrics for AI Employee {self.employee_id}: {str(e)}")
    
    async def log_operation(self, operation: str, data: Dict[str, Any], result: Dict[str, Any]):
        """Log an operation performed by the AI Employee."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "data": data,
                "result": result,
                "status": self.status.status
            }
            
            self.operation_log.append(log_entry)
            
            # Update counters
            self.status.total_operations += 1
            if result.get("success", False):
                self.status.success_count += 1
            else:
                self.status.error_count += 1
            
            self.status.last_activity = datetime.now()
            
        except Exception as e:
            logger.error(f"Error logging operation for AI Employee {self.employee_id}: {str(e)}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration for this AI Employee."""
        try:
            # Get role-specific configuration
            role_config = settings.ai_employee_roles.get(self.spec.role, {})
            
            # Merge with spec configuration
            config = {
                "model_architecture": self.spec.model_architecture,
                "training_data_sources": self.spec.training_data_sources,
                "evaluation_metrics": self.spec.evaluation_metrics,
                "optimization_focus": self.spec.optimization_focus,
                "learning_rate": self.spec.learning_rate,
                "batch_size": self.spec.batch_size,
                "epochs": self.spec.epochs,
                "role_specific": role_config
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading config for AI Employee {self.employee_id}: {str(e)}")
            return {}
    
    async def _cleanup(self):
        """Cleanup resources when stopping the AI Employee."""
        try:
            # Clear model from memory
            if self.model is not None:
                del self.model
                self.model = None
            
            # Clear performance history if too large
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
            
            # Clear operation log if too large
            if len(self.operation_log) > 500:
                self.operation_log = self.operation_log[-250:]
                
        except Exception as e:
            logger.error(f"Error during cleanup for AI Employee {self.employee_id}: {str(e)}")
    
    async def _validate_task_data(self, task_data: Dict[str, Any]) -> bool:
        """Validate task data before execution."""
        try:
            required_fields = self._get_required_task_fields()
            
            for field in required_fields:
                if field not in task_data:
                    logger.error(f"Missing required field '{field}' in task data")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating task data: {str(e)}")
            return False
    
    def _get_required_task_fields(self) -> List[str]:
        """Get required fields for task data. Override in subclasses."""
        return []
    
    async def _preprocess_data(self, data: Any) -> Any:
        """Preprocess data before use. Override in subclasses."""
        return data
    
    async def _postprocess_result(self, result: Any) -> Any:
        """Postprocess result before returning. Override in subclasses."""
        return result
    
    def _calculate_accuracy(self, predictions: List[Any], targets: List[Any]) -> float:
        """Calculate accuracy metric."""
        try:
            if len(predictions) != len(targets):
                return 0.0
            
            correct = sum(1 for p, t in zip(predictions, targets) if p == t)
            return correct / len(predictions) if predictions else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {str(e)}")
            return 0.0
    
    def _calculate_execution_speed(self, start_time: datetime, end_time: datetime) -> float:
        """Calculate execution speed in seconds."""
        try:
            duration = (end_time - start_time).total_seconds()
            return duration
        except Exception as e:
            logger.error(f"Error calculating execution speed: {str(e)}")
            return 0.0
    
    async def _handle_error(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Handle errors during operations."""
        try:
            error_info = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "operation": operation,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log the error
            logger.error(f"Error in {operation} for AI Employee {self.employee_id}: {str(error)}")
            
            # Update status
            self.status.error_count += 1
            
            return {
                "success": False,
                "error": error_info,
                "employee_id": self.employee_id
            }
            
        except Exception as e:
            logger.error(f"Error handling error: {str(e)}")
            return {
                "success": False,
                "error": {"error_message": "Unknown error occurred"},
                "employee_id": self.employee_id
            }
    
    def __str__(self) -> str:
        """String representation of the AI Employee."""
        return f"{self.__class__.__name__}(id={self.employee_id}, role={self.spec.role}, status={self.status.status})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the AI Employee."""
        return f"{self.__class__.__name__}(employee_id='{self.employee_id}', role='{self.spec.role}', specialization='{self.spec.specialization}', status='{self.status.status}')" 