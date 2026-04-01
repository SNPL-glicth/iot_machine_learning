"""Concrete IoT tools for ZENIN system.

Initial tools:
- SendAlertTool: Send notifications to operators
- AdjustThresholdTool: Dynamically adjust sensor thresholds

Safety rules implemented:
- CRITICAL severity → AUTO
- WARNING severity → ASK  
- Low confidence (< 0.6) → ASK
- Too frequent adjustments → DENY
"""

from __future__ import annotations

from typing import Any, Dict

from .tool import Tool, ToolContext, ToolResult, GuardResult, SafetyLevel


class SendAlertTool(Tool):
    """Send alert notification to device operators.
    
    Safety:
    - CRITICAL: auto-approve (immediate action needed)
    - WARNING: ask for approval
    - Normal: auto-approve (informational)
    """
    
    @property
    def name(self) -> str:
        return "send_alert"
    
    @property
    def description(self) -> str:
        return "Send alert notification to device operator via configured channels"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "critical"],
                    "description": "Alert severity level"
                },
                "message": {
                    "type": "string",
                    "description": "Alert message content"
                },
                "channels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Notification channels (email, sms, push, webhook)",
                    "default": ["push"]
                },
                "device_id": {
                    "type": "integer",
                    "description": "Target device ID"
                }
            },
            "required": ["severity", "message"]
        }
    
    def can_execute(self, context: ToolContext) -> GuardResult:
        """Evaluate safety for sending alert."""
        # CRITICAL: always auto-approve
        if context.severity.upper() == "CRITICAL":
            return GuardResult.allow("Critical severity requires immediate notification")
        
        # WARNING: ask for approval (important but not urgent)
        if context.severity.upper() == "WARNING":
            return GuardResult.ask("Warning severity notifications require approval")
        
        # Normal/Info: auto-approve (informational only)
        return GuardResult.allow("Informational alert")
    
    def execute(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute alert sending (placeholder implementation).
        
        In production, this would integrate with:
        - Email service
        - SMS gateway
        - Push notification service
        - Webhook endpoints
        """
        severity = params.get("severity", "info")
        message = params.get("message", "")
        channels = params.get("channels", ["push"])
        device_id = params.get("device_id", context.device_id)
        
        # Placeholder: simulate successful sending
        # TODO: Integrate with actual notification services
        sent_to = []
        for channel in channels:
            # Simulate channel send
            sent_to.append(f"{channel}:{device_id}")
        
        return ToolResult.ok(
            output={
                "sent": True,
                "channels_used": channels,
                "device_id": device_id,
                "severity": severity,
                "message_preview": message[:50] + "..." if len(message) > 50 else message,
            },
            metadata={
                "alert_type": severity,
                "target_channels": channels,
                "series_id": context.series_id,
            }
        )


class AdjustThresholdTool(Tool):
    """Dynamically adjust sensor alert threshold.
    
    Safety:
    - CRITICAL device + invasive change → ASK
    - Low confidence (< 0.6) → ASK
    - Too frequent adjustments (>3 in 1 hour) → DENY
    - Otherwise → AUTO
    """
    
    # Rate limiting: max adjustments per hour per device
    _MAX_ADJUSTMENTS_PER_HOUR = 3
    
    @property
    def name(self) -> str:
        return "adjust_threshold"
    
    @property
    def description(self) -> str:
        return "Adjust sensor alert threshold based on recent patterns"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "new_threshold": {
                    "type": "number",
                    "description": "New threshold value"
                },
                "threshold_type": {
                    "type": "string",
                    "enum": ["min", "max", "both"],
                    "description": "Which threshold to adjust",
                    "default": "max"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for adjustment"
                }
            },
            "required": ["new_threshold"]
        }
    
    def can_execute(self, context: ToolContext) -> GuardResult:
        """Evaluate safety for threshold adjustment."""
        # Low confidence: ask for approval
        if context.confidence < 0.6:
            return GuardResult.ask(
                f"Low confidence ({context.confidence:.2f}) requires approval for threshold changes"
            )
        
        # Rate limiting: check recent adjustments
        recent_adjustments = [
            a for a in context.historical_actions
            if a.get("tool") == "adjust_threshold"
            and a.get("time_since_seconds", float('inf')) < 3600  # 1 hour
        ]
        
        if len(recent_adjustments) >= self._MAX_ADJUSTMENTS_PER_HOUR:
            return GuardResult.deny(
                f"Too many recent threshold adjustments ({len(recent_adjustments)} in last hour). "
                f"Rate limit: {self._MAX_ADJUSTMENTS_PER_HOUR}/hour"
            )
        
        # Critical device: ask for approval (threshold changes are invasive)
        if context.severity.upper() == "CRITICAL":
            return GuardResult.ask(
                "Critical device threshold changes require approval"
            )
        
        # All checks passed: auto-approve
        return GuardResult.allow(
            f"Within safe adjustment limits. Recent adjustments: {len(recent_adjustments)}/hour"
        )
    
    def execute(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute threshold adjustment (placeholder).
        
        In production, this would:
        - Validate new threshold against safety bounds
        - Update threshold in database
        - Trigger recalibration of related predictions
        - Log change for audit
        """
        new_threshold = params.get("new_threshold")
        threshold_type = params.get("threshold_type", "max")
        reason = params.get("reason", "Automated adjustment based on pattern analysis")
        
        # Placeholder: simulate successful adjustment
        # TODO: Integrate with actual threshold storage
        
        return ToolResult.ok(
            output={
                "adjusted": True,
                "new_threshold": new_threshold,
                "threshold_type": threshold_type,
                "series_id": context.series_id,
                "device_id": context.device_id,
                "previous_value": context.current_value,  # For reference
            },
            metadata={
                "adjustment_reason": reason,
                "confidence_at_adjustment": context.confidence,
                "requires_audit": True,
            }
        )


class RequestMaintenanceTool(Tool):
    """Create maintenance ticket/request.
    
    Safety:
    - Always ASK (maintenance is expensive/disruptive)
    """
    
    @property
    def name(self) -> str:
        return "request_maintenance"
    
    @property
    def description(self) -> str:
        return "Create maintenance ticket for device inspection or repair"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                    "description": "Maintenance priority",
                    "default": "medium"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for maintenance request"
                },
                "estimated_downtime_minutes": {
                    "type": "integer",
                    "description": "Estimated downtime if known",
                    "default": 0
                }
            },
            "required": ["reason"]
        }
    
    def can_execute(self, context: ToolContext) -> GuardResult:
        """Maintenance always requires approval."""
        return GuardResult.ask(
            "Maintenance requests require human approval (disruptive operation)"
        )
    
    def execute(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute maintenance request (placeholder)."""
        priority = params.get("priority", "medium")
        reason = params.get("reason", "")
        downtime = params.get("estimated_downtime_minutes", 0)
        
        # Placeholder: simulate ticket creation
        # TODO: Integrate with ticketing system (Jira, ServiceNow, etc.)
        
        return ToolResult.ok(
            output={
                "ticket_created": True,
                "ticket_id": f"MAINT-{context.device_id}-{int(context.time_since_last_action_seconds())}",
                "priority": priority,
                "device_id": context.device_id,
                "estimated_downtime_minutes": downtime,
            },
            metadata={
                "maintenance_reason": reason,
                "requires_scheduling": True,
                "approval_status": "approved"  # Since we auto-approved ASK level
            }
        )
