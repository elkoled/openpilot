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
from typing import Iterable, Mapping, MutableMapping, Sequence

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
    self._last_snapshot_summary: str | None = None

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

  def _annotate_process_issues(self, states: Sequence[MutableMapping[str, object]]):
    issues_by_type: dict[str, set[str]] = {}
    non_zero_exit: dict[str, int] = {}
    should_run_but_not: list[str] = []
    running_but_should_not: list[str] = []
    running_while_disabled: list[str] = []
    sigkill_pending: list[str] = []
    ignored_processes: list[str] = []
    multiple_starts: list[dict[str, int]] = []
    last_stop_sigkill: list[str] = []
    issue_details: dict[str, list[str]] = {}

    for state in states:
      name = str(state.get("name", "<unknown>"))
      running = bool(state.get("running", False))
      should_run_val = state.get("last_should_run")
      if should_run_val is None:
        should_run_val = state.get("shouldBeRunning")
      should_run = bool(should_run_val) if should_run_val is not None else None
      enabled_val = state.get("enabled")
      enabled = bool(enabled_val) if enabled_val is not None else True

      exit_code_val = state.get("exit_code")
      if exit_code_val is None:
        exit_code_val = state.get("exitCode")
      try:
        exit_code = int(exit_code_val) if exit_code_val is not None else None
      except (TypeError, ValueError):
        exit_code = None

      sigkill = bool(state.get("sigkill") or state.get("sigKill"))
      shutting_down = bool(state.get("shutting_down") or state.get("shuttingDown"))
      last_stop_signal = state.get("last_stop_signal") or state.get("lastStopSignal")
      start_count_val = state.get("start_count") if "start_count" in state else state.get("startCount")
      try:
        start_count = int(start_count_val) if start_count_val is not None else None
      except (TypeError, ValueError):
        start_count = None
      ignored = bool(state.get("ignored", False))

      issues: list[str] = []

      if should_run is True and not running:
        issues.append("should_run_but_not_running")
        should_run_but_not.append(name)
      if should_run is False and running:
        issues.append("running_when_should_not")
        running_but_should_not.append(name)
      if not enabled and running:
        issues.append("running_while_disabled")
        running_while_disabled.append(name)
      if exit_code not in (None, 0):
        issues.append("non_zero_exit")
        non_zero_exit[name] = exit_code
      if sigkill:
        issues.append("sigkill_pending")
        sigkill_pending.append(name)
      if shutting_down and running:
        issues.append("in_shutdown")
      if isinstance(start_count, int) and start_count > 1:
        issues.append("multiple_starts")
        multiple_starts.append({"name": name, "start_count": start_count})
      if ignored:
        issues.append("ignored_process")
        ignored_processes.append(name)
      if isinstance(last_stop_signal, str) and last_stop_signal.upper() == "SIGKILL":
        issues.append("last_stop_was_sigkill")
        last_stop_sigkill.append(name)

      state["issues"] = issues

      if issues:
        issue_details[name] = issues
        for issue in issues:
          issues_by_type.setdefault(issue, set()).add(name)

    summary = {
      "total_processes": len(states),
      "total_issues": sum(len(issues) for issues in issue_details.values()),
      "processes_with_issues": sorted(issue_details.keys()),
      "issues_by_type": {issue: sorted(names) for issue, names in sorted(issues_by_type.items())},
      "non_zero_exit": dict(sorted(non_zero_exit.items())),
      "should_run_but_not_running": sorted(set(should_run_but_not)),
      "running_when_should_not": sorted(set(running_but_should_not)),
      "running_while_disabled": sorted(set(running_while_disabled)),
      "sigkill_pending": sorted(set(sigkill_pending)),
      "last_stop_was_sigkill": sorted(set(last_stop_sigkill)),
      "ignored_processes": sorted(set(ignored_processes)),
      "multiple_starts": sorted(multiple_starts, key=lambda item: item["name"]),
      "issue_details": {name: issues for name, issues in sorted(issue_details.items())},
    }

    return summary

  def log_process_snapshot(self, processes) -> None:
    snapshot: list[MutableMapping[str, object]] = []
    for name in sorted(processes.keys()):
      proc = processes[name]
      state: MutableMapping[str, object] = {
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

    summary = self._annotate_process_issues(snapshot)

    payload = {
      "processes": snapshot,
      "summary": summary,
    }

    serialized = json.dumps(payload, sort_keys=True, default=str)
    if serialized != self._last_snapshot:
      self._last_snapshot = serialized
      self._logger.debug("process_snapshot %s", serialized)

    summary_serialized = json.dumps(summary, sort_keys=True, default=str)
    if summary.get("total_issues", 0):
      if summary_serialized != self._last_snapshot_summary:
        self._last_snapshot_summary = summary_serialized
        self._logger.warning("process_snapshot_issues | %s", summary_serialized)
    else:
      self._last_snapshot_summary = None

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
    manager_processes_list: list[MutableMapping[str, object]] = [dict(proc) for proc in manager_processes]
    manager_summary = self._annotate_process_issues(manager_processes_list)

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
      "manager_processes": manager_processes_list,
      "manager_process_summary": manager_summary,
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
