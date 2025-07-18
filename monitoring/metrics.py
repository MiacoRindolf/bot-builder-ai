"""
Metrics collection and monitoring system for the Bot Builder AI system.
Tracks AI Employee performance, system health, and operational metrics.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for AI Employees."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    execution_speed: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class SystemMetrics:
    """System-level metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: float = 0.0
    active_connections: int = 0
    error_count: int = 0
    request_count: int = 0
    response_time_avg: float = 0.0
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class MetricsCollector:
    """
    Metrics collector for tracking AI Employee performance and system health.
    
    Responsibilities:
    - Collect and store performance metrics
    - Monitor system health
    - Generate performance reports
    - Track operational statistics
    - Provide real-time monitoring
    """
    
    def __init__(self):
        """Initialize the Metrics Collector."""
        self.metrics_storage = {}
        self.performance_history = {}
        self.system_metrics = SystemMetrics()
        self.employee_status = {}
        
        # Create metrics directory
        self.metrics_dir = Path("monitoring/metrics_data")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics configuration
        self.metrics_retention_days = 30
        self.update_interval = 60  # seconds
        
        # Start background monitoring
        self.monitoring_task = None
        self.is_monitoring = False
        
        logger.info("Metrics Collector initialized")
    
    async def start_monitoring(self):
        """Start background monitoring."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Background monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            logger.info("Background monitoring stopped")
    
    async def update_employee_metrics(self, employee_id: str, metrics: Dict[str, Any]):
        """
        Update metrics for a specific AI Employee.
        
        Args:
            employee_id: AI Employee ID
            metrics: Performance metrics dictionary
        """
        try:
            # Convert to PerformanceMetrics object
            performance_metrics = PerformanceMetrics(**metrics)
            
            # Store current metrics
            self.metrics_storage[employee_id] = performance_metrics
            
            # Add to history
            if employee_id not in self.performance_history:
                self.performance_history[employee_id] = []
            
            # Convert datetime to string for JSON serialization
            metrics_dict = asdict(performance_metrics)
            metrics_dict['last_updated'] = metrics_dict['last_updated'].isoformat()
            
            self.performance_history[employee_id].append({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics_dict
            })
            
            # Keep only recent history
            cutoff_date = datetime.now() - timedelta(days=self.metrics_retention_days)
            self.performance_history[employee_id] = [
                entry for entry in self.performance_history[employee_id]
                if datetime.fromisoformat(entry['timestamp']) > cutoff_date
            ]
            
            # Save to file
            await self._save_employee_metrics(employee_id, performance_metrics)
            
            logger.debug(f"Updated metrics for employee {employee_id}")
            
        except Exception as e:
            logger.error(f"Error updating metrics for employee {employee_id}: {str(e)}")
    
    async def update_employee_status(self, employee_id: str, status: str):
        """
        Update status for a specific AI Employee.
        
        Args:
            employee_id: AI Employee ID
            status: New status
        """
        try:
            self.employee_status[employee_id] = {
                'status': status,
                'last_updated': datetime.now().isoformat()
            }
            
            # Save to file
            await self._save_employee_status(employee_id, status)
            
            logger.debug(f"Updated status for employee {employee_id}: {status}")
            
        except Exception as e:
            logger.error(f"Error updating status for employee {employee_id}: {str(e)}")
    
    async def get_employee_metrics(self, employee_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific AI Employee.
        
        Args:
            employee_id: AI Employee ID
            
        Returns:
            Dictionary with employee metrics
        """
        try:
            if employee_id in self.metrics_storage:
                metrics = self.metrics_storage[employee_id]
                metrics_dict = asdict(metrics)
                # Convert datetime to string for JSON serialization
                if metrics_dict.get('last_updated'):
                    metrics_dict['last_updated'] = metrics_dict['last_updated'].isoformat()
                return metrics_dict
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting metrics for employee {employee_id}: {str(e)}")
            return {}
    
    async def get_employee_status(self, employee_id: str) -> str:
        """
        Get status for a specific AI Employee.
        
        Args:
            employee_id: AI Employee ID
            
        Returns:
            Employee status string
        """
        try:
            if employee_id in self.employee_status:
                return self.employee_status[employee_id]['status']
            else:
                return "unknown"
                
        except Exception as e:
            logger.error(f"Error getting status for employee {employee_id}: {str(e)}")
            return "unknown"
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for all AI Employees.
        
        Returns:
            Dictionary with all employee metrics
        """
        try:
            all_metrics = {}
            for employee_id in self.metrics_storage:
                all_metrics[employee_id] = asdict(self.metrics_storage[employee_id])
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error getting all metrics: {str(e)}")
            return {}
    
    async def get_all_performance(self) -> Dict[str, Any]:
        """
        Get performance data for all AI Employees.
        
        Returns:
            Dictionary with performance data
        """
        try:
            performance_data = {}
            
            for employee_id, history in self.performance_history.items():
                if history:
                    latest = history[-1]
                    performance_data[employee_id] = {
                        'current_metrics': latest['metrics'],
                        'status': await self.get_employee_status(employee_id),
                        'history_count': len(history),
                        'last_updated': latest['timestamp']
                    }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting all performance: {str(e)}")
            return {}
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            Dictionary with system metrics
        """
        try:
            return asdict(self.system_metrics)
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary across all AI Employees.
        
        Returns:
            Dictionary with performance summary
        """
        try:
            if not self.metrics_storage:
                return {}
            
            # Calculate aggregate metrics
            accuracies = [m.accuracy for m in self.metrics_storage.values()]
            speeds = [m.execution_speed for m in self.metrics_storage.values()]
            success_rates = [m.success_rate for m in self.metrics_storage.values()]
            
            summary = {
                'total_employees': len(self.metrics_storage),
                'average_accuracy': statistics.mean(accuracies) if accuracies else 0.0,
                'average_execution_speed': statistics.mean(speeds) if speeds else 0.0,
                'average_success_rate': statistics.mean(success_rates) if success_rates else 0.0,
                'best_performer': self._get_best_performer(),
                'worst_performer': self._get_worst_performer(),
                'system_health': self._assess_system_health()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {}
    
    async def get_employee_performance(self, employee_id: str) -> Dict[str, Any]:
        """
        Get detailed performance data for an AI Employee.
        
        Args:
            employee_id: AI Employee ID
            
        Returns:
            Dictionary with detailed performance data
        """
        try:
            if employee_id not in self.performance_history:
                return {}
            
            history = self.performance_history[employee_id]
            if not history:
                return {}
            
            # Calculate trends
            recent_metrics = history[-10:] if len(history) >= 10 else history
            
            accuracy_trend = [entry['metrics']['accuracy'] for entry in recent_metrics]
            speed_trend = [entry['metrics']['execution_speed'] for entry in recent_metrics]
            
            performance_data = {
                'employee_id': employee_id,
                'current_metrics': history[-1]['metrics'],
                'status': await self.get_employee_status(employee_id),
                'trends': {
                    'accuracy_trend': accuracy_trend,
                    'speed_trend': speed_trend,
                    'accuracy_change': self._calculate_change(accuracy_trend),
                    'speed_change': self._calculate_change(speed_trend)
                },
                'history': history,
                'recommendations': self._generate_recommendations(history[-1]['metrics'])
            }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting employee performance for {employee_id}: {str(e)}")
            return {}
    
    async def record_operation(self, employee_id: str, operation: str, duration: float, success: bool):
        """
        Record an operation for metrics tracking.
        
        Args:
            employee_id: AI Employee ID
            operation: Operation type
            duration: Operation duration in seconds
            success: Whether operation was successful
        """
        try:
            if employee_id not in self.metrics_storage:
                self.metrics_storage[employee_id] = PerformanceMetrics()
            
            metrics = self.metrics_storage[employee_id]
            
            # Update execution speed (rolling average)
            if metrics.execution_speed == 0.0:
                metrics.execution_speed = duration
            else:
                metrics.execution_speed = (metrics.execution_speed * 0.9) + (duration * 0.1)
            
            # Update success rate
            if success:
                metrics.success_rate = min(1.0, metrics.success_rate + 0.01)
            else:
                metrics.success_rate = max(0.0, metrics.success_rate - 0.01)
            
            # Update error rate
            if not success:
                metrics.error_rate = min(1.0, metrics.error_rate + 0.01)
            else:
                metrics.error_rate = max(0.0, metrics.error_rate - 0.005)
            
            metrics.last_updated = datetime.now()
            
            # Update stored metrics
            await self.update_employee_metrics(employee_id, asdict(metrics))
            
        except Exception as e:
            logger.error(f"Error recording operation for {employee_id}: {str(e)}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Update system metrics
                await self._update_system_metrics()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Save metrics to disk
                await self._save_all_metrics()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.update_interval)
    
    async def _update_system_metrics(self):
        """Update system-level metrics."""
        try:
            # Simulate system metrics collection
            self.system_metrics.cpu_usage = self._get_cpu_usage()
            self.system_metrics.memory_usage = self._get_memory_usage()
            self.system_metrics.disk_usage = self._get_disk_usage()
            self.system_metrics.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {str(e)}")
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        # Placeholder implementation
        import random
        return random.uniform(20.0, 80.0)
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        # Placeholder implementation
        import random
        return random.uniform(30.0, 90.0)
    
    def _get_disk_usage(self) -> float:
        """Get disk usage percentage."""
        # Placeholder implementation
        import random
        return random.uniform(10.0, 70.0)
    
    async def _cleanup_old_data(self):
        """Clean up old metrics data."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.metrics_retention_days)
            
            for employee_id in list(self.performance_history.keys()):
                self.performance_history[employee_id] = [
                    entry for entry in self.performance_history[employee_id]
                    if datetime.fromisoformat(entry['timestamp']) > cutoff_date
                ]
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
    
    async def _save_employee_metrics(self, employee_id: str, metrics: PerformanceMetrics):
        """Save employee metrics to file."""
        try:
            file_path = self.metrics_dir / f"metrics_{employee_id}.json"
            
            data = {
                'employee_id': employee_id,
                'metrics': asdict(metrics),
                'saved_at': datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics for {employee_id}: {str(e)}")
    
    async def _save_employee_status(self, employee_id: str, status: str):
        """Save employee status to file."""
        try:
            file_path = self.metrics_dir / f"status_{employee_id}.json"
            
            data = {
                'employee_id': employee_id,
                'status': status,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving status for {employee_id}: {str(e)}")
    
    async def _save_all_metrics(self):
        """Save all metrics to disk."""
        try:
            # Save system metrics
            system_file = self.metrics_dir / "system_metrics.json"
            system_metrics_dict = asdict(self.system_metrics)
            # Convert datetime to string for JSON serialization
            if system_metrics_dict.get('last_updated'):
                system_metrics_dict['last_updated'] = system_metrics_dict['last_updated'].isoformat()
            with open(system_file, 'w') as f:
                json.dump(system_metrics_dict, f, indent=2)
            
            # Save employee statuses
            status_file = self.metrics_dir / "employee_statuses.json"
            with open(status_file, 'w') as f:
                json.dump(self.employee_status, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving all metrics: {str(e)}")
    
    def _get_best_performer(self) -> Optional[str]:
        """Get the best performing AI Employee."""
        try:
            if not self.metrics_storage:
                return None
            
            best_employee = max(
                self.metrics_storage.items(),
                key=lambda x: x[1].accuracy * x[1].success_rate
            )
            
            return best_employee[0]
            
        except Exception as e:
            logger.error(f"Error getting best performer: {str(e)}")
            return None
    
    def _get_worst_performer(self) -> Optional[str]:
        """Get the worst performing AI Employee."""
        try:
            if not self.metrics_storage:
                return None
            
            worst_employee = min(
                self.metrics_storage.items(),
                key=lambda x: x[1].accuracy * x[1].success_rate
            )
            
            return worst_employee[0]
            
        except Exception as e:
            logger.error(f"Error getting worst performer: {str(e)}")
            return None
    
    def get_api_usage_history(self) -> List[int]:
        """Get API usage history for the last 24 hours."""
        try:
            # Return mock data for now - in production this would come from actual API logs
            return [45, 52, 38, 29, 23, 18, 25, 67, 89, 124, 156, 178, 
                    145, 167, 189, 201, 234, 267, 289, 312, 298, 245, 189, 156]
        except Exception as e:
            logger.error(f"Error getting API usage history: {str(e)}")
            return [0] * 24
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends for the last 30 days."""
        try:
            # Return mock data for now - in production this would come from actual metrics
            dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, 0, -1)]
            success_rates = [85 + i * 0.3 for i in range(30)]
            response_times = [1.5 - i * 0.01 for i in range(30)]
            
            return {
                "dates": dates,
                "success_rates": success_rates,
                "response_times": response_times
            }
        except Exception as e:
            logger.error(f"Error getting performance trends: {str(e)}")
            return {"dates": [], "success_rates": [], "response_times": []}
    
    def _assess_system_health(self) -> str:
        """Assess overall system health."""
        try:
            if not self.metrics_storage:
                return "unknown"
            
            # Calculate average metrics
            avg_accuracy = statistics.mean([m.accuracy for m in self.metrics_storage.values()])
            avg_success_rate = statistics.mean([m.success_rate for m in self.metrics_storage.values()])
            avg_error_rate = statistics.mean([m.error_rate for m in self.metrics_storage.values()])
            
            # Assess health based on metrics
            if avg_accuracy > 0.8 and avg_success_rate > 0.9 and avg_error_rate < 0.1:
                return "excellent"
            elif avg_accuracy > 0.7 and avg_success_rate > 0.8 and avg_error_rate < 0.2:
                return "good"
            elif avg_accuracy > 0.6 and avg_success_rate > 0.7 and avg_error_rate < 0.3:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Error assessing system health: {str(e)}")
            return "unknown"
    
    def _calculate_change(self, values: List[float]) -> float:
        """Calculate percentage change in a list of values."""
        try:
            if len(values) < 2:
                return 0.0
            
            first_value = values[0]
            last_value = values[-1]
            
            if first_value == 0:
                return 0.0
            
            return ((last_value - first_value) / first_value) * 100
            
        except Exception as e:
            logger.error(f"Error calculating change: {str(e)}")
            return 0.0
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        try:
            accuracy = metrics.get('accuracy', 0.0)
            success_rate = metrics.get('success_rate', 0.0)
            execution_speed = metrics.get('execution_speed', 0.0)
            error_rate = metrics.get('error_rate', 0.0)
            
            if accuracy < 0.7:
                recommendations.append("Consider retraining the model with more diverse data")
            
            if success_rate < 0.8:
                recommendations.append("Review error patterns and improve error handling")
            
            if execution_speed > 1.0:
                recommendations.append("Optimize execution speed through model optimization")
            
            if error_rate > 0.2:
                recommendations.append("Investigate high error rate and implement fixes")
            
            if not recommendations:
                recommendations.append("Performance is good, continue monitoring")
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("Unable to generate recommendations")
        
        return recommendations
    
    async def export_metrics(self, format: str = "json") -> str:
        """
        Export all metrics in specified format.
        
        Args:
            format: Export format (json, csv)
            
        Returns:
            Exported data as string
        """
        try:
            if format.lower() == "json":
                # Convert system metrics with datetime handling
                system_metrics_dict = asdict(self.system_metrics)
                if system_metrics_dict.get('last_updated'):
                    system_metrics_dict['last_updated'] = system_metrics_dict['last_updated'].isoformat()
                
                # Convert employee metrics with datetime handling
                employee_metrics_dict = {}
                for k, v in self.metrics_storage.items():
                    metrics_dict = asdict(v)
                    if metrics_dict.get('last_updated'):
                        metrics_dict['last_updated'] = metrics_dict['last_updated'].isoformat()
                    employee_metrics_dict[k] = metrics_dict
                
                return json.dumps({
                    'system_metrics': system_metrics_dict,
                    'employee_metrics': employee_metrics_dict,
                    'employee_status': self.employee_status,
                    'exported_at': datetime.now().isoformat()
                }, indent=2)
            else:
                return "Unsupported export format"
                
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
            return "Error exporting metrics"
    
    async def reset_metrics(self, employee_id: Optional[str] = None):
        """
        Reset metrics for an employee or all employees.
        
        Args:
            employee_id: Specific employee ID or None for all
        """
        try:
            if employee_id:
                if employee_id in self.metrics_storage:
                    del self.metrics_storage[employee_id]
                if employee_id in self.performance_history:
                    del self.performance_history[employee_id]
                if employee_id in self.employee_status:
                    del self.employee_status[employee_id]
                logger.info(f"Reset metrics for employee {employee_id}")
            else:
                self.metrics_storage.clear()
                self.performance_history.clear()
                self.employee_status.clear()
                logger.info("Reset all metrics")
                
        except Exception as e:
            logger.error(f"Error resetting metrics: {str(e)}") 