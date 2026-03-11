"""
Alerting Module
===============
Handles alert rules and notifications for monitoring events.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.config import get_monitoring_config


class AlertManager:
    """Manages monitoring alerts and notifications."""

    def __init__(self):
        config = get_monitoring_config()
        self.alert_config = config.get("alerts", {})
        self.enabled = self.alert_config.get("enabled", True)
        self.channels = self.alert_config.get("channels", [])
        self.alert_history: list[dict] = []

    def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "WARNING",
        details: dict | None = None,
    ) -> None:
        """Send an alert through configured channels.

        Args:
            alert_type: Type of alert (drift, performance, error).
            message: Alert message.
            severity: Alert severity level.
            details: Additional alert details.
        """
        if not self.enabled:
            return

        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message,
            "severity": severity,
            "details": details or {},
        }

        self.alert_history.append(alert)

        for channel in self.channels:
            if channel["type"] == "log":
                self._send_log_alert(alert, channel.get("level", "WARNING"))
            elif channel["type"] == "file":
                self._send_file_alert(alert, channel.get("path", "reports/alerts.json"))

    def _send_log_alert(self, alert: dict, level: str) -> None:
        """Send alert to logger."""
        msg = f"🚨 [{alert['type'].upper()}] {alert['message']}"
        getattr(logger, level.lower(), logger.warning)(msg)

    def _send_file_alert(self, alert: dict, path: str) -> None:
        """Append alert to a JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        alerts = []
        if filepath.exists():
            with open(filepath, "r") as f:
                try:
                    alerts = json.load(f)
                except json.JSONDecodeError:
                    alerts = []

        alerts.append(alert)

        with open(filepath, "w") as f:
            json.dump(alerts, f, indent=2)

    def get_recent_alerts(self, n: int = 10) -> list[dict]:
        """Get the N most recent alerts."""
        return self.alert_history[-n:]
