"""
Approval Engine - Human-in-the-Loop Approval Workflow Management
Handles approval workflows, notifications, and audit trails for self-improvement proposals.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from config.settings import settings

logger = logging.getLogger(__name__)

class ApprovalStatus(Enum):
    """Approval status enumeration."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"

class ApprovalPriority(Enum):
    """Approval priority levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class ApprovalRequest:
    """An approval request for a self-improvement proposal."""
    id: str
    proposal_id: str
    title: str
    description: str
    rationale: str
    risk_level: str
    priority: ApprovalPriority
    estimated_impact: Dict[str, Any]
    code_changes_summary: Dict[str, Any]
    requested_by: str
    created_at: datetime
    expires_at: datetime
    status: ApprovalStatus
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class ApprovalNotification:
    """Notification for approval requests."""
    id: str
    approval_request_id: str
    type: str  # "NEW_REQUEST", "APPROVED", "REJECTED", "EXPIRED"
    title: str
    message: str
    priority: str
    created_at: datetime
    read: bool = False
    action_required: bool = False

class ApprovalEngine:
    """
    Human-in-the-Loop Approval Engine.
    
    Responsibilities:
    - Manage approval workflows
    - Send notifications
    - Track approval history
    - Handle approval timeouts
    - Maintain audit trails
    """
    
    def __init__(self):
        """Initialize the Approval Engine."""
        # Approval tracking
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        self.approval_history: List[ApprovalRequest] = []
        self.notifications: List[ApprovalNotification] = []
        
        # Configuration
        self.default_expiry_hours = 24
        self.auto_approve_low_risk = True
        self.auto_approve_threshold = 0.3  # Risk threshold for auto-approval
        
        # Callbacks for external systems
        self.notification_callbacks: List[Callable] = []
        self.approval_callbacks: List[Callable] = []
        
        # Audit trail
        self.audit_log: List[Dict[str, Any]] = []
        
        logger.info("Approval Engine initialized successfully")
    
    async def initialize(self) -> bool:
        """Initialize the Approval Engine."""
        try:
            # Start background tasks
            asyncio.create_task(self._cleanup_expired_approvals())
            asyncio.create_task(self._process_auto_approvals())
            
            logger.info("Approval Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Approval Engine: {str(e)}")
            return False
    
    async def submit_proposal(self, proposal: Any) -> bool:
        """
        Submit a proposal for approval.
        
        Args:
            proposal: SelfImprovementProposal object
            
        Returns:
            True if submitted successfully, False otherwise
        """
        try:
            # Check if auto-approval is possible
            if self._should_auto_approve(proposal):
                logger.info(f"Auto-approving proposal {proposal.id} (low risk)")
                return await self._auto_approve_proposal(proposal)
            
            # Create approval request
            approval_request = ApprovalRequest(
                id=str(uuid.uuid4()),
                proposal_id=proposal.id,
                title=proposal.title,
                description=proposal.description,
                rationale=proposal.rationale,
                risk_level=proposal.risk_level,
                priority=ApprovalPriority(proposal.priority),
                estimated_impact=proposal.estimated_impact,
                code_changes_summary=self._summarize_code_changes(proposal.code_changes),
                requested_by=proposal.generated_by,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=self.default_expiry_hours),
                status=ApprovalStatus.PENDING
            )
            
            # Store the approval request
            self.pending_approvals[approval_request.id] = approval_request
            
            # Create notification
            notification = self._create_notification(approval_request, "NEW_REQUEST")
            self.notifications.append(notification)
            
            # Log audit trail
            self._log_audit_event("PROPOSAL_SUBMITTED", {
                "approval_id": approval_request.id,
                "proposal_id": proposal.id,
                "title": proposal.title,
                "risk_level": proposal.risk_level
            })
            
            # Trigger callbacks
            await self._trigger_notification_callbacks(notification)
            
            logger.info(f"Submitted proposal {proposal.id} for approval: {approval_request.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting proposal: {str(e)}")
            return False
    
    async def approve_proposal(self, approval_id: str, approved_by: str, 
                             notes: Optional[str] = None) -> bool:
        """
        Approve a proposal.
        
        Args:
            approval_id: ID of the approval request
            approved_by: User who approved
            notes: Optional notes
            
        Returns:
            True if approved successfully, False otherwise
        """
        try:
            if approval_id not in self.pending_approvals:
                logger.error(f"Approval request {approval_id} not found")
                return False
            
            approval_request = self.pending_approvals[approval_id]
            
            # Update approval request
            approval_request.status = ApprovalStatus.APPROVED
            approval_request.approved_by = approved_by
            approval_request.approved_at = datetime.now()
            approval_request.notes = notes
            
            # Move to history
            self.approval_history.append(approval_request)
            del self.pending_approvals[approval_id]
            
            # Create notification
            notification = self._create_notification(approval_request, "APPROVED")
            self.notifications.append(notification)
            
            # Log audit trail
            self._log_audit_event("PROPOSAL_APPROVED", {
                "approval_id": approval_id,
                "proposal_id": approval_request.proposal_id,
                "approved_by": approved_by,
                "notes": notes
            })
            
            # Trigger callbacks
            await self._trigger_approval_callbacks(approval_request)
            await self._trigger_notification_callbacks(notification)
            
            logger.info(f"Approved proposal {approval_request.proposal_id} by {approved_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error approving proposal: {str(e)}")
            return False
    
    async def reject_proposal(self, approval_id: str, rejected_by: str, 
                            reason: str, notes: Optional[str] = None) -> bool:
        """
        Reject a proposal.
        
        Args:
            approval_id: ID of the approval request
            rejected_by: User who rejected
            reason: Reason for rejection
            notes: Optional notes
            
        Returns:
            True if rejected successfully, False otherwise
        """
        try:
            if approval_id not in self.pending_approvals:
                logger.error(f"Approval request {approval_id} not found")
                return False
            
            approval_request = self.pending_approvals[approval_id]
            
            # Update approval request
            approval_request.status = ApprovalStatus.REJECTED
            approval_request.approved_by = rejected_by
            approval_request.approved_at = datetime.now()
            approval_request.rejection_reason = reason
            approval_request.notes = notes
            
            # Move to history
            self.approval_history.append(approval_request)
            del self.pending_approvals[approval_id]
            
            # Create notification
            notification = self._create_notification(approval_request, "REJECTED")
            self.notifications.append(notification)
            
            # Log audit trail
            self._log_audit_event("PROPOSAL_REJECTED", {
                "approval_id": approval_id,
                "proposal_id": approval_request.proposal_id,
                "rejected_by": rejected_by,
                "reason": reason,
                "notes": notes
            })
            
            # Trigger callbacks
            await self._trigger_notification_callbacks(notification)
            
            logger.info(f"Rejected proposal {approval_request.proposal_id} by {rejected_by}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error rejecting proposal: {str(e)}")
            return False
    
    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return list(self.pending_approvals.values())
    
    def get_approval_history(self, limit: int = 50) -> List[ApprovalRequest]:
        """Get recent approval history."""
        return self.approval_history[-limit:] if self.approval_history else []
    
    def get_notifications(self, unread_only: bool = False) -> List[ApprovalNotification]:
        """Get notifications."""
        notifications = self.notifications
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        return notifications
    
    def mark_notification_read(self, notification_id: str) -> bool:
        """Mark a notification as read."""
        try:
            for notification in self.notifications:
                if notification.id == notification_id:
                    notification.read = True
                    return True
            return False
        except Exception as e:
            logger.error(f"Error marking notification as read: {str(e)}")
            return False
    
    def add_notification_callback(self, callback: Callable):
        """Add a notification callback."""
        self.notification_callbacks.append(callback)
    
    def add_approval_callback(self, callback: Callable):
        """Add an approval callback."""
        self.approval_callbacks.append(callback)
    
    def get_approval_stats(self) -> Dict[str, Any]:
        """Get approval statistics."""
        try:
            total_requests = len(self.approval_history) + len(self.pending_approvals)
            approved_requests = len([r for r in self.approval_history if r.status == ApprovalStatus.APPROVED])
            rejected_requests = len([r for r in self.approval_history if r.status == ApprovalStatus.REJECTED])
            pending_requests = len(self.pending_approvals)
            
            approval_rate = approved_requests / max(1, total_requests)
            
            return {
                "total_requests": total_requests,
                "approved_requests": approved_requests,
                "rejected_requests": rejected_requests,
                "pending_requests": pending_requests,
                "approval_rate": approval_rate,
                "average_approval_time_hours": self._calculate_average_approval_time()
            }
            
        except Exception as e:
            logger.error(f"Error calculating approval stats: {str(e)}")
            return {"error": str(e)}
    
    def _should_auto_approve(self, proposal: Any) -> bool:
        """Check if a proposal should be auto-approved."""
        if not self.auto_approve_low_risk:
            return False
        
        # Auto-approve low-risk, low-priority proposals
        risk_levels = {"LOW": 0.1, "MEDIUM": 0.5, "HIGH": 0.9}
        risk_score = risk_levels.get(proposal.risk_level, 0.5)
        
        return (risk_score <= self.auto_approve_threshold and 
                proposal.priority in ["LOW", "MEDIUM"])
    
    async def _auto_approve_proposal(self, proposal: Any) -> bool:
        """Auto-approve a proposal."""
        try:
            # Create approval request for audit trail
            approval_request = ApprovalRequest(
                id=str(uuid.uuid4()),
                proposal_id=proposal.id,
                title=proposal.title,
                description=proposal.description,
                rationale=proposal.rationale,
                risk_level=proposal.risk_level,
                priority=ApprovalPriority(proposal.priority),
                estimated_impact=proposal.estimated_impact,
                code_changes_summary=self._summarize_code_changes(proposal.code_changes),
                requested_by=proposal.generated_by,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1),
                status=ApprovalStatus.APPROVED,
                approved_by="AUTO_APPROVAL_SYSTEM",
                approved_at=datetime.now(),
                notes="Auto-approved due to low risk"
            )
            
            # Add to history
            self.approval_history.append(approval_request)
            
            # Log audit trail
            self._log_audit_event("PROPOSAL_AUTO_APPROVED", {
                "approval_id": approval_request.id,
                "proposal_id": proposal.id,
                "title": proposal.title,
                "reason": "Low risk auto-approval"
            })
            
            # Trigger callbacks
            await self._trigger_approval_callbacks(approval_request)
            
            logger.info(f"Auto-approved proposal {proposal.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error auto-approving proposal: {str(e)}")
            return False
    
    def _summarize_code_changes(self, code_changes: Dict[str, str]) -> Dict[str, Any]:
        """Create a summary of code changes."""
        try:
            summary = {
                "total_files": len(code_changes),
                "files_modified": list(code_changes.keys()),
                "total_changes": 0,
                "change_types": {"additions": 0, "deletions": 0, "modifications": 0}
            }
            
            for file_path, diff in code_changes.items():
                lines = diff.split('\n')
                additions = len([line for line in lines if line.startswith('+') and not line.startswith('+++')])
                deletions = len([line for line in lines if line.startswith('-') and not line.startswith('---')])
                
                summary["total_changes"] += additions + deletions
                summary["change_types"]["additions"] += additions
                summary["change_types"]["deletions"] += deletions
                
                if additions > 0 and deletions > 0:
                    summary["change_types"]["modifications"] += 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing code changes: {str(e)}")
            return {"error": str(e)}
    
    def _create_notification(self, approval_request: ApprovalRequest, 
                           notification_type: str) -> ApprovalNotification:
        """Create a notification for an approval request."""
        notification_id = str(uuid.uuid4())
        
        if notification_type == "NEW_REQUEST":
            title = f"New Approval Request: {approval_request.title}"
            message = f"A new self-improvement proposal requires your approval. Priority: {approval_request.priority.value}"
            action_required = True
        elif notification_type == "APPROVED":
            title = f"Proposal Approved: {approval_request.title}"
            message = f"The proposal has been approved by {approval_request.approved_by}"
            action_required = False
        elif notification_type == "REJECTED":
            title = f"Proposal Rejected: {approval_request.title}"
            message = f"The proposal has been rejected. Reason: {approval_request.rejection_reason}"
            action_required = False
        else:
            title = f"Approval Update: {approval_request.title}"
            message = f"Status update for approval request"
            action_required = False
        
        return ApprovalNotification(
            id=notification_id,
            approval_request_id=approval_request.id,
            type=notification_type,
            title=title,
            message=message,
            priority=approval_request.priority.value,
            created_at=datetime.now(),
            action_required=action_required
        )
    
    async def _cleanup_expired_approvals(self):
        """Background task to cleanup expired approvals."""
        while True:
            try:
                current_time = datetime.now()
                expired_approvals = []
                
                for approval_id, approval_request in self.pending_approvals.items():
                    if current_time > approval_request.expires_at:
                        expired_approvals.append(approval_id)
                
                for approval_id in expired_approvals:
                    approval_request = self.pending_approvals[approval_id]
                    approval_request.status = ApprovalStatus.EXPIRED
                    
                    # Move to history
                    self.approval_history.append(approval_request)
                    del self.pending_approvals[approval_id]
                    
                    # Create notification
                    notification = self._create_notification(approval_request, "EXPIRED")
                    self.notifications.append(notification)
                    
                    # Log audit trail
                    self._log_audit_event("PROPOSAL_EXPIRED", {
                        "approval_id": approval_id,
                        "proposal_id": approval_request.proposal_id
                    })
                    
                    logger.info(f"Expired approval request: {approval_id}")
                
                # Wait for next cleanup cycle
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup expired approvals: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _process_auto_approvals(self):
        """Background task to process auto-approvals."""
        while True:
            try:
                # This would check for proposals that meet auto-approval criteria
                # For now, just a placeholder
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in auto-approval processing: {str(e)}")
                await asyncio.sleep(300)
    
    async def _trigger_notification_callbacks(self, notification: ApprovalNotification):
        """Trigger notification callbacks."""
        for callback in self.notification_callbacks:
            try:
                await callback(notification)
            except Exception as e:
                logger.error(f"Error in notification callback: {str(e)}")
    
    async def _trigger_approval_callbacks(self, approval_request: ApprovalRequest):
        """Trigger approval callbacks."""
        for callback in self.approval_callbacks:
            try:
                await callback(approval_request)
            except Exception as e:
                logger.error(f"Error in approval callback: {str(e)}")
    
    def _log_audit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log an audit event."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "event_data": event_data
        }
        self.audit_log.append(audit_entry)
    
    def _calculate_average_approval_time(self) -> float:
        """Calculate average approval time in hours."""
        try:
            approved_requests = [r for r in self.approval_history 
                               if r.status == ApprovalStatus.APPROVED and r.approved_at]
            
            if not approved_requests:
                return 0.0
            
            total_time = sum(
                (r.approved_at - r.created_at).total_seconds() / 3600
                for r in approved_requests
            )
            
            return total_time / len(approved_requests)
            
        except Exception as e:
            logger.error(f"Error calculating average approval time: {str(e)}")
            return 0.0 