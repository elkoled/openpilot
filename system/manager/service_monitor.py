"""Utility helpers for monitoring the openpilot manager service.

This module centralizes the additional logging required to debug
manager lifecycle events when openpilot is off-road.  The logger writes
structured information to a rotating file on disk without modifying the
manager control flow.
"""
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from openpilot.system.hardware.hw import Paths


class ServiceMonitor:
  """Collects diagnostic information about the manager service.

  The monitor mirrors existing telemetry without influencing the
  execution flow.  It persists the output to a rotating log file so the
  data remains available even when the standard on-road logging is
  inactive.
  """

  def __init__(self,
               log_filename: str = "openpilot_service_monitor.log",
               max_bytes: int = 5 * 1024 * 1024,
               backup_count: int = 5) -> None:
    log_dir = Path(Paths.swaglog_root())
    log_dir.mkdir(parents=True, exist_ok=True)

    self._log_path = log_dir / log_filename
    self._logger = logging.getLogger("openpilot.service_monitor")
    self._logger.setLevel(logging.DEBUG)
    self._logger.propagate = False

    handler_exists = any(
      isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", None) == str(self._log_path)
      for handler in self._logger.handlers
    )

    if not handler_exists:
      handler = RotatingFileHandler(self._log_path, maxBytes=max_bytes, backupCount=backup_count)
      handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
      ))
      self._logger.addHandler(handler)

    self._last_snapshot: str | None = None

  def log_event(self, message: str, **payload: object) -> None:
    if payload:
      serialized = json.dumps(payload, sort_keys=True, default=str)
      self._logger.info("%s | %s", message, serialized)
    else:
      self._logger.info(message)

  def log_manager_start(self, *, ignore: Iterable[str], environ: dict[str, str]) -> None:
    self.log_event(
      "manager_thread_start",
      ignore=list(ignore),
      environ={k: environ[k] for k in sorted(environ)},
    )

  def log_manager_init(self, *, serial: str, version: str, branch: str, commit: str, dirty: bool) -> None:
    self.log_event(
      "manager_init",
      serial=serial,
      version=version,
      branch=branch,
      commit=commit,
      dirty=dirty,
      log_file=str(self._log_path),
    )

  def log_process_snapshot(self, processes) -> None:
    snapshot = []
    for name in sorted(processes.keys()):
      proc = processes[name]
      state = {
        "name": name,
        "running": bool(proc.proc and proc.proc.is_alive()),
        "pid": getattr(proc.proc, "pid", None),
        "exit_code": getattr(proc.proc, "exitcode", None),
        "shutting_down": getattr(proc, "shutting_down", False),
        "enabled": getattr(proc, "enabled", False),
        "type": proc.__class__.__name__,
        "supports_stacktrace": getattr(proc, "supports_stacktrace", False),
        "sigkill": getattr(proc, "sigkill", False),
        "daemon": getattr(proc, "daemon", False),
        "watchdog_max_dt": getattr(proc, "watchdog_max_dt", None),
        "start_count": getattr(proc, "start_count", None),
        "last_start_time": getattr(proc, "last_start_time", None),
        "last_start_monotonic": getattr(proc, "last_start_monotonic", None),
        "last_should_run": getattr(proc, "last_should_run", None),
        "last_ensure_started": getattr(proc, "last_ensure_started", None),
        "last_ensure_time": getattr(proc, "last_ensure_time", None),
        "last_watchdog_checked": getattr(proc, "last_watchdog_checked", None),
        "last_stop_signal": getattr(proc, "last_stop_signal", None),
        "last_stop_signal_time": getattr(proc, "last_stop_signal_time", None),
      }
      if hasattr(proc, "module"):
        state["module"] = proc.module
      if hasattr(proc, "cmdline"):
        state["cmdline"] = proc.cmdline
      if hasattr(proc, "param_name"):
        state["param_name"] = proc.param_name
      if hasattr(proc, "daemon_pid"):
        state["daemon_pid"] = getattr(proc, "daemon_pid", None)
      snapshot.append(state)

    serialized = json.dumps(snapshot, sort_keys=True, default=str)
    if serialized != self._last_snapshot:
      self._last_snapshot = serialized
      self._logger.debug("process_snapshot %s", serialized)

  def log_process_start(self, *, name: str, pid: int | None) -> None:
    self.log_event("process_start", name=name, pid=pid)

  def log_process_exit(self,
                       *,
                       name: str,
                       pid: int | None,
                       exit_code: int | None,
                       shutting_down: bool,
                       restart: bool | None = None) -> None:
    self.log_event(
      "process_exit",
      name=name,
      pid=pid,
      exit_code=exit_code,
      shutting_down=shutting_down,
      restart=restart,
    )

  def log_process_exception(self, *, name: str, stacktrace: str) -> None:
    self.log_event("process_exception", name=name, stacktrace=stacktrace)

  def log_process_stacktrace(self, *, name: str, trigger: str, stacktrace: str) -> None:
    self.log_event("process_stacktrace", name=name, trigger=trigger, stacktrace=stacktrace)

  def log_stacktrace_request(self, *, name: str, pid: int, signal: str, reason: str | None = None) -> None:
    self.log_event("process_stacktrace_requested", name=name, pid=pid, signal=signal, reason=reason)

  def log_watchdog_timeout(self, *, name: str, elapsed: float, exit_code: int | None) -> None:
    self.log_event("watchdog_timeout", name=name, elapsed=elapsed, exit_code=exit_code)

  def log_process_restart(self, *, name: str, reason: str | None = None) -> None:
    self.log_event("process_restart", name=name, reason=reason)

  def log_shutdown_request(self, param: str) -> None:
    self.log_event("shutdown_requested", reason=param)

  def log_manager_cleanup(self) -> None:
    self.log_event("manager_cleanup")

  def log_exception(self, context: str) -> None:
    self._logger.exception("%s", context)

  def log_comm_issue(self,
                     *,
                     reason: str,
                     status: Mapping[str, Sequence[str]],
                     frame: int,
                     recv_frame: Mapping[str, int],
                     frame_age: Mapping[str, int | None],
                     valid: Mapping[str, bool],
                     alive: Mapping[str, bool],
                     freq_ok: Mapping[str, bool],
                     updated: Mapping[str, bool],
                     manager_processes: Sequence[Mapping[str, object]],
                     events: Sequence[Mapping[str, object]],
                     not_running: Sequence[str],
                     ignored_processes: Sequence[str],
                     ratekeeper: Mapping[str, object],
                     extra: Mapping[str, object] | None = None) -> None:
    payload: dict[str, object] = {
      "reason": reason,
      "status": {k: sorted(v) for k, v in status.items()},
      "frame": frame,
      "recv_frame": dict(sorted(recv_frame.items())),
      "frame_age": dict(sorted(frame_age.items())),
      "valid": dict(sorted(valid.items())),
      "alive": dict(sorted(alive.items())),
      "freq_ok": dict(sorted(freq_ok.items())),
      "updated": dict(sorted(updated.items())),
      "manager_processes": list(manager_processes),
      "events": list(events),
      "not_running": sorted(not_running),
      "ignored_processes": sorted(ignored_processes),
      "ratekeeper": dict(ratekeeper),
    }

    if extra:
      payload["extra"] = dict(extra)

    serialized = json.dumps(payload, sort_keys=True, default=str)
    self._logger.warning("comm_issue_detected | %s", serialized)


service_monitor = ServiceMonitor()

__all__ = ["ServiceMonitor", "service_monitor"]
