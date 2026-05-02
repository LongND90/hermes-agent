"""
Owner Approval rate limiter + denylist for unauthorized-DM "notify_owner" mode.

Tracks per-user notification cooldowns, a global per-minute cap, and a
persistent denylist of users the owner has explicitly blocked. State is
stored under ``~/.hermes/pairing/`` with chmod 0600 for parity with the
existing pairing store.
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_dir, get_hermes_home
from utils import atomic_replace

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

PAIRING_DIR = get_hermes_dir("platforms/pairing", "pairing")

DEFAULT_COOLDOWN_SECONDS = 600
DEFAULT_PER_MINUTE_GLOBAL = 5
DEFAULT_AUTO_BLOCK_AFTER_DENY = 3


def _secure_write(path: Path, data: str) -> None:
    """Atomic write with chmod 0600 (mirrors gateway.pairing._secure_write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_owner_approval_config() -> Dict[str, Any]:
    """Read ``gateway.owner_approval`` block from ``~/.hermes/config.yaml``.

    Returns an empty dict on any failure — defaults are applied by callers.
    """
    if _yaml is None:
        return {}
    cfg_path = get_hermes_home() / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    gateway = data.get("gateway") or {}
    if not isinstance(gateway, dict):
        return {}
    block = gateway.get("owner_approval") or {}
    return block if isinstance(block, dict) else {}


class OwnerApprovalRateLimiter:
    """Throttles owner notifications and persists per-user deny counters + denylist."""

    def __init__(self, platform: str = "telegram"):
        self.platform = platform
        self._lock = threading.RLock()
        cfg = _load_owner_approval_config()
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.cooldown_seconds: int = int(cfg.get("notify_cooldown_seconds", DEFAULT_COOLDOWN_SECONDS))
        self.per_minute_global: int = int(cfg.get("notify_per_minute_global", DEFAULT_PER_MINUTE_GLOBAL))
        self.auto_block_after_deny: int = int(cfg.get("auto_block_after_deny_count", DEFAULT_AUTO_BLOCK_AFTER_DENY))
        PAIRING_DIR.mkdir(parents=True, exist_ok=True)

    # --- Paths ---

    def _denied_path(self) -> Path:
        return PAIRING_DIR / f"{self.platform}-denied.json"

    def _state_path(self) -> Path:
        return PAIRING_DIR / "_owner_notify_state.json"

    # --- IO helpers ---

    def _load(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, path: Path, data: dict) -> None:
        _secure_write(path, json.dumps(data, indent=2, ensure_ascii=False))

    # --- Denylist ---

    def is_blocked(self, user_id: str) -> bool:
        with self._lock:
            denied = self._load(self._denied_path())
            return str(user_id) in denied

    def block(self, user_id: str, blocked_by: Optional[str] = None) -> None:
        with self._lock:
            denied = self._load(self._denied_path())
            denied[str(user_id)] = {
                "blocked_at": time.time(),
                "blocked_by_owner_id": str(blocked_by) if blocked_by else None,
            }
            self._save(self._denied_path(), denied)
            # Clear any deny counter for this user once they're blocked.
            state = self._load(self._state_path())
            state.pop(f"deny:{user_id}", None)
            self._save(self._state_path(), state)

    def unblock(self, user_id: str) -> bool:
        with self._lock:
            denied = self._load(self._denied_path())
            if str(user_id) in denied:
                del denied[str(user_id)]
                self._save(self._denied_path(), denied)
                return True
        return False

    # --- Notification rate limit ---

    def should_notify(self, user_id: str) -> bool:
        """Return True if owner can be notified about this user right now.

        Blocks when:
          - feature disabled
          - user is on the denylist
          - per-user cooldown not elapsed
          - global notifications-per-minute cap exceeded
        """
        if not self.enabled:
            return False
        if self.is_blocked(user_id):
            return False
        with self._lock:
            state = self._load(self._state_path())
            now = time.time()
            user_key = f"user:{user_id}"
            last = float(state.get(user_key, 0) or 0)
            if (now - last) < self.cooldown_seconds:
                return False
            recent = [t for t in state.get("global", []) if (now - float(t)) < 60.0]
            if len(recent) >= self.per_minute_global:
                return False
        return True

    def record_notification(self, user_id: str) -> None:
        with self._lock:
            state = self._load(self._state_path())
            now = time.time()
            state[f"user:{user_id}"] = now
            recent = [float(t) for t in state.get("global", []) if (now - float(t)) < 60.0]
            recent.append(now)
            state["global"] = recent
            self._save(self._state_path(), state)

    # --- Deny counter (auto-block after N denies) ---

    def record_deny(self, user_id: str) -> int:
        """Increment deny counter for a user and return new count."""
        with self._lock:
            state = self._load(self._state_path())
            key = f"deny:{user_id}"
            count = int(state.get(key, 0) or 0) + 1
            state[key] = count
            self._save(self._state_path(), state)
        return count

    def reset_deny(self, user_id: str) -> None:
        with self._lock:
            state = self._load(self._state_path())
            if state.pop(f"deny:{user_id}", None) is not None:
                self._save(self._state_path(), state)
