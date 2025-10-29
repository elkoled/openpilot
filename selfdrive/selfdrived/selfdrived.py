#!/usr/bin/env python3
import os
import signal
import time
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cereal.messaging as messaging

from cereal import car, log, custom
from msgq.visionipc import VisionIpcClient, VisionStreamType


from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, Priority, Ratekeeper, DT_CTRL
from openpilot.common.swaglog import cloudlog
from openpilot.common.gps import get_gps_location_service

from openpilot.selfdrive.car.car_specific import CarSpecificEvents
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose
from openpilot.selfdrive.selfdrived.events import Events, ET
from openpilot.selfdrive.selfdrived.helpers import ExcessiveActuationCheck
from openpilot.selfdrive.selfdrived.state import StateMachine
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager, set_offroad_alert

from openpilot.system.manager.process_config import STACKTRACE_CAPABLE_PROCESSES
from openpilot.system.manager.service_monitor import service_monitor
from openpilot.system.version import get_build_metadata

from openpilot.sunnypilot.mads.mads import ModularAssistiveDrivingSystem
from openpilot.sunnypilot import get_sanitize_int_param
from openpilot.sunnypilot.selfdrive.car.car_specific import CarSpecificEventsSP
from openpilot.sunnypilot.selfdrive.car.cruise_helpers import CruiseHelper
from openpilot.sunnypilot.selfdrive.car.intelligent_cruise_button_management.controller import IntelligentCruiseButtonManagement
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP

REPLAY = "REPLAY" in os.environ
SIMULATION = "SIMULATION" in os.environ
TESTING_CLOSET = "TESTING_CLOSET" in os.environ

LONGITUDINAL_PERSONALITY_MAP = {v: k for k, v in log.LongitudinalPersonality.schema.enumerants.items()}
STACKTRACE_CAPABLE_PROCESSES_SET = frozenset(STACKTRACE_CAPABLE_PROCESSES)


def _safe_read_text(path: Path) -> Optional[str]:
  try:
    return path.read_text(errors='replace')
  except (FileNotFoundError, PermissionError, OSError):
    return None


def _read_proc_cmdline(pid: int) -> Optional[List[str]]:
  try:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
  except (FileNotFoundError, PermissionError, OSError):
    return None

  parts = [segment.decode(errors='replace') for segment in raw.split(b"\0") if segment]
  return parts


def _read_proc_key_value(pid: int, name: str, keys: Optional[Tuple[str, ...]] = None) -> Optional[Dict[str, str]]:
  raw = _safe_read_text(Path(f"/proc/{pid}/{name}"))
  if raw is None:
    return None

  result: Dict[str, str] = {}
  for line in raw.splitlines():
    key, _, value = line.partition(":")
    if not key:
      continue
    key = key.strip()
    if keys is None or key in keys:
      result[key] = value.strip()
  return result


def _read_proc_link(pid: int, name: str) -> Optional[str]:
  try:
    return os.readlink(f"/proc/{pid}/{name}")
  except (FileNotFoundError, PermissionError, OSError):
    return None


def _count_open_fds(pid: int) -> Optional[int]:
  fd_dir = Path(f"/proc/{pid}/fd")
  try:
    return sum(1 for _ in fd_dir.iterdir())
  except (FileNotFoundError, PermissionError, OSError):
    return None


def _collect_process_diagnostics(proc_state) -> Dict[str, Any]:
  pid = int(proc_state.pid)
  diag: Dict[str, Any] = {
    "name": proc_state.name,
    "pid": pid,
    "running": bool(proc_state.running),
    "shouldBeRunning": bool(proc_state.shouldBeRunning),
    "exitCode": int(proc_state.exitCode),
    "supportsStacktrace": proc_state.name in STACKTRACE_CAPABLE_PROCESSES_SET,
  }

  if pid > 0:
    cmdline = _read_proc_cmdline(pid)
    if cmdline is not None:
      diag["cmdline"] = cmdline

    status_keys = ("Name", "State", "Threads", "VmRSS", "VmSize", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")
    status = _read_proc_key_value(pid, "status", status_keys)
    if status is not None:
      diag["status"] = status

    io_keys = ("rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes")
    io_stats = _read_proc_key_value(pid, "io", io_keys)
    if io_stats is not None:
      diag["io"] = io_stats

    diag["cwd"] = _read_proc_link(pid, "cwd")
    diag["exe"] = _read_proc_link(pid, "exe")
    diag["fd_count"] = _count_open_fds(pid)

  return diag


def _summarize_panda_state(ps) -> Dict[str, Any]:
  summary: Dict[str, Any] = {
    'pandaType': int(ps.pandaType),
    'ignitionLine': bool(ps.ignitionLine),
    'ignitionCan': bool(ps.ignitionCan),
    'voltage': float(ps.voltage),
    'current': float(ps.current),
    'safetyModel': int(ps.safetyModel),
    'safetyParam': int(ps.safetyParam),
    'alternativeExperience': int(ps.alternativeExperience),
  }
  try:
    summary['faults'] = [int(fault) for fault in ps.faults]
  except Exception:
    summary['faults'] = []
  return summary

ThermalStatus = log.DeviceState.ThermalStatus
State = log.SelfdriveState.OpenpilotState
PandaType = log.PandaState.PandaType
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection
EventName = log.OnroadEvent.EventName
ButtonType = car.CarState.ButtonEvent.Type
SafetyModel = car.CarParams.SafetyModel
TurnDirection = custom.ModelDataV2SP.TurnDirection

IGNORED_SAFETY_MODES = (SafetyModel.silent, SafetyModel.noOutput)


class SelfdriveD(CruiseHelper):
  def __init__(self, CP=None, CP_SP=None):
    self.params = Params()

    # Ensure the current branch is cached, otherwise the first cycle lags
    build_metadata = get_build_metadata()

    if CP is None:
      cloudlog.info("selfdrived is waiting for CarParams")
      self.CP = messaging.log_from_bytes(self.params.get("CarParams", block=True), car.CarParams)
      cloudlog.info("selfdrived got CarParams")
    else:
      self.CP = CP

    if CP_SP is None:
      cloudlog.info("selfdrived is waiting for CarParamsSP")
      self.CP_SP = messaging.log_from_bytes(self.params.get("CarParamsSP", block=True), custom.CarParamsSP)
      cloudlog.info("selfdrived got CarParamsSP")
    else:
      self.CP_SP = CP_SP

    self.car_events = CarSpecificEvents(self.CP)

    self.pose_calibrator = PoseCalibrator()
    self.calibrated_pose: Pose | None = None
    self.excessive_actuation_check = ExcessiveActuationCheck()
    self.excessive_actuation = self.params.get("Offroad_ExcessiveActuation") is not None

    # Setup sockets
    self.pm = messaging.PubMaster(['selfdriveState', 'onroadEvents'] + ['selfdriveStateSP', 'onroadEventsSP'])

    self.gps_location_service = get_gps_location_service(self.params)
    self.gps_packets = [self.gps_location_service]
    self.sensor_packets = ["accelerometer", "gyroscope"]
    self.camera_packets = ["roadCameraState", "driverCameraState", "wideRoadCameraState"]

    # TODO: de-couple selfdrived with card/conflate on carState without introducing controls mismatches
    self.car_state_sock = messaging.sub_sock('carState', timeout=20)

    ignore = self.sensor_packets + self.gps_packets + ['alertDebug'] + ['modelDataV2SP']
    if SIMULATION:
      ignore += ['driverCameraState', 'managerState']
    if REPLAY:
      # no vipc in replay will make them ignored anyways
      ignore += ['roadCameraState', 'wideRoadCameraState']
    self.sm = messaging.SubMaster(['deviceState', 'pandaStates', 'peripheralState', 'modelV2', 'liveCalibration',
                                   'carOutput', 'driverMonitoringState', 'longitudinalPlan', 'livePose', 'liveDelay',
                                   'managerState', 'liveParameters', 'radarState', 'liveTorqueParameters',
                                   'controlsState', 'carControl', 'driverAssistance', 'alertDebug', 'userBookmark', 'audioFeedback',
                                   'modelDataV2SP', 'longitudinalPlanSP'] + \
                                   self.camera_packets + self.sensor_packets + self.gps_packets,
                                  ignore_alive=ignore, ignore_avg_freq=ignore,
                                  ignore_valid=ignore, frequency=int(1/DT_CTRL))

    # read params
    self.is_metric = self.params.get_bool("IsMetric")
    self.is_ldw_enabled = self.params.get_bool("IsLdwEnabled")
    self.disengage_on_accelerator = self.params.get_bool("DisengageOnAccelerator")

    car_recognized = self.CP.brand != 'mock'

    # cleanup old params
    if not self.CP.alphaLongitudinalAvailable:
      self.params.remove("AlphaLongitudinalEnabled")
    if not self.CP.openpilotLongitudinalControl:
      self.params.remove("ExperimentalMode")

    self.CS_prev = car.CarState.new_message()
    self.AM = AlertManager()
    self.events = Events()

    self.initialized = False
    self.enabled = False
    self.active = False
    self.mismatch_counter = 0
    self.cruise_mismatch_counter = 0
    self.last_steering_pressed_frame = 0
    self.distance_traveled = 0
    self.last_functional_fan_frame = 0
    self.events_prev = []
    self.logged_comm_issue = None
    self.comm_issue_active = False
    self.stacktrace_request_context: Dict[str, Tuple[int, str]] = {}
    self.self_stacktrace_token: Optional[Tuple[int, str]] = None
    self.last_comm_issue_reason: Optional[str] = None
    self.not_running_prev = None
    self.experimental_mode = False
    self.personality = get_sanitize_int_param(
      "LongitudinalPersonality",
      min(log.LongitudinalPersonality.schema.enumerants.values()),
      max(log.LongitudinalPersonality.schema.enumerants.values()),
      self.params
    )
    self.recalibrating_seen = False
    self.state_machine = StateMachine()
    self.rk = Ratekeeper(100, print_delay_threshold=None)

    self.ignored_processes = {'mapd', }

    # Determine startup event
    is_remote = build_metadata.openpilot.comma_remote or build_metadata.openpilot.sunnypilot_remote
    self.startup_event = EventName.startup if is_remote and build_metadata.tested_channel else EventName.startupMaster
    if not car_recognized:
      self.startup_event = EventName.startupNoCar
    elif car_recognized and self.CP.passive:
      self.startup_event = EventName.startupNoControl
    elif self.CP.secOcRequired and not self.CP.secOcKeyAvailable:
      self.startup_event = EventName.startupNoSecOcKey

    if not car_recognized:
      self.events.add(EventName.carUnrecognized, static=True)
      set_offroad_alert("Offroad_CarUnrecognized", True)
    elif self.CP.passive:
      self.events.add(EventName.dashcamMode, static=True)

    self.events_sp = EventsSP()
    self.events_sp_prev = []

    self.mads = ModularAssistiveDrivingSystem(self)
    self.icbm = IntelligentCruiseButtonManagement(self.CP, self.CP_SP)

    self.car_events_sp = CarSpecificEventsSP(self.CP, self.CP_SP)

    CruiseHelper.__init__(self, self.CP)

  def update_events(self, CS):
    """Compute onroadEvents from carState"""

    self.events.clear()
    self.events_sp.clear()

    if self.sm['controlsState'].lateralControlState.which() == 'debugState':
      self.events.add(EventName.joystickDebug)
      self.startup_event = None

    if self.sm.recv_frame['alertDebug'] > 0:
      self.events.add(EventName.longitudinalManeuver)
      self.startup_event = None

    # Add startup event
    if self.startup_event is not None:
      self.events.add(self.startup_event)
      self.startup_event = None

    # Don't add any more events if not initialized
    if not self.initialized:
      self.events.add(EventName.selfdriveInitializing)
      return

    # Check for user bookmark press (bookmark button or end of LKAS button feedback)
    if self.sm.updated['userBookmark']:
      self.events.add(EventName.userBookmark)

    if self.sm.updated['audioFeedback']:
      self.events.add(EventName.audioFeedback)

    # Don't add any more events while in dashcam mode
    if self.CP.passive:
      return

    # Block resume if cruise never previously enabled
    resume_pressed = any(be.type in (ButtonType.accelCruise, ButtonType.resumeCruise) for be in CS.buttonEvents)
    if not self.CP.pcmCruise and CS.vCruise > 250 and resume_pressed:
      self.events.add(EventName.resumeBlocked)

    if not self.CP.notCar:
      self.events.add_from_msg(self.sm['driverMonitoringState'].events)
      self.events_sp.add_from_msg(self.sm['longitudinalPlanSP'].events)

    # Add car events, ignore if CAN isn't valid
    if CS.canValid:
      car_events = self.car_events.update(CS, self.CS_prev, self.sm['carControl']).to_msg()
      self.events.add_from_msg(car_events)

      car_events_sp = self.car_events_sp.update(CS, self.events).to_msg()
      self.events_sp.add_from_msg(car_events_sp)

      if self.CP.notCar:
        # wait for everything to init first
        if self.sm.frame > int(5. / DT_CTRL) and self.initialized:
          # body always wants to enable
          self.events.add(EventName.pcmEnable)

      # Disable on rising edge of accelerator or brake. Also disable on brake when speed > 0
      if (CS.gasPressed and not self.CS_prev.gasPressed and self.disengage_on_accelerator) or \
        (CS.brakePressed and (not self.CS_prev.brakePressed or not CS.standstill)) or \
        (CS.regenBraking and (not self.CS_prev.regenBraking or not CS.standstill)):
        self.events.add(EventName.pedalPressed)

    # Create events for temperature, disk space, and memory
    if self.sm['deviceState'].thermalStatus >= ThermalStatus.red:
      self.events.add(EventName.overheat)
    if self.sm['deviceState'].freeSpacePercent < 7 and not SIMULATION:
      self.events.add(EventName.outOfSpace)
    if self.sm['deviceState'].memoryUsagePercent > 90 and not SIMULATION:
      self.events.add(EventName.lowMemory)

    # Alert if fan isn't spinning for 5 seconds
    if self.sm['peripheralState'].pandaType != log.PandaState.PandaType.unknown:
      if self.sm['peripheralState'].fanSpeedRpm < 500 and self.sm['deviceState'].fanSpeedPercentDesired > 50:
        # allow enough time for the fan controller in the panda to recover from stalls
        if (self.sm.frame - self.last_functional_fan_frame) * DT_CTRL > 15.0:
          self.events.add(EventName.fanMalfunction)
      else:
        self.last_functional_fan_frame = self.sm.frame

    # Handle calibration status
    cal_status = self.sm['liveCalibration'].calStatus
    if cal_status != log.LiveCalibrationData.Status.calibrated:
      if cal_status == log.LiveCalibrationData.Status.uncalibrated:
        self.events.add(EventName.calibrationIncomplete)
      elif cal_status == log.LiveCalibrationData.Status.recalibrating:
        if not self.recalibrating_seen:
          set_offroad_alert("Offroad_Recalibration", True)
        self.recalibrating_seen = True
        self.events.add(EventName.calibrationRecalibrating)
      else:
        self.events.add(EventName.calibrationInvalid)

    # Lane departure warning
    if self.is_ldw_enabled and self.sm.valid['driverAssistance']:
      if self.sm['driverAssistance'].leftLaneDeparture or self.sm['driverAssistance'].rightLaneDeparture:
        self.events.add(EventName.ldw)

    # ******************************************************************************************
    #  NOTE: To fork maintainers.
    #  Disabling or nerfing safety features will get you and your users banned from our servers.
    #  We recommend that you do not change these numbers from the defaults.
    if self.sm.updated['liveCalibration']:
      self.pose_calibrator.feed_live_calib(self.sm['liveCalibration'])
    if self.sm.updated['livePose']:
      device_pose = Pose.from_live_pose(self.sm['livePose'])
      self.calibrated_pose = self.pose_calibrator.build_calibrated_pose(device_pose)

    if self.calibrated_pose is not None:
      excessive_actuation = self.excessive_actuation_check.update(self.sm, CS, self.calibrated_pose)
      if not self.excessive_actuation and excessive_actuation is not None:
        set_offroad_alert("Offroad_ExcessiveActuation", True, extra_text=str(excessive_actuation))
        self.excessive_actuation = True

    if self.excessive_actuation:
      self.events.add(EventName.excessiveActuation)
    # ******************************************************************************************

    # Handle lane change
    if self.sm['modelV2'].meta.laneChangeState == LaneChangeState.preLaneChange:
      direction = self.sm['modelV2'].meta.laneChangeDirection
      if (CS.leftBlindspot and direction == LaneChangeDirection.left) or \
         (CS.rightBlindspot and direction == LaneChangeDirection.right):
        self.events.add(EventName.laneChangeBlocked)
      else:
        if direction == LaneChangeDirection.left:
          self.events.add(EventName.preLaneChangeLeft)
        else:
          self.events.add(EventName.preLaneChangeRight)
    elif self.sm['modelV2'].meta.laneChangeState in (LaneChangeState.laneChangeStarting,
                                                    LaneChangeState.laneChangeFinishing):
      self.events.add(EventName.laneChange)

    # Handle lane turn
    lane_turn_direction = self.sm['modelDataV2SP'].laneTurnDirection
    if lane_turn_direction == TurnDirection.turnLeft:
      self.events_sp.add(custom.OnroadEventSP.EventName.laneTurnLeft)
    elif lane_turn_direction == TurnDirection.turnRight:
      self.events_sp.add(custom.OnroadEventSP.EventName.laneTurnRight)

    for i, pandaState in enumerate(self.sm['pandaStates']):
      # All pandas must match the list of safetyConfigs, and if outside this list, must be silent or noOutput
      if i < len(self.CP.safetyConfigs):
        safety_mismatch = pandaState.safetyModel != self.CP.safetyConfigs[i].safetyModel or \
                          pandaState.safetyParam != self.CP.safetyConfigs[i].safetyParam or \
                          pandaState.alternativeExperience != self.CP.alternativeExperience
      else:
        safety_mismatch = pandaState.safetyModel not in IGNORED_SAFETY_MODES

      # safety mismatch allows some time for pandad to set the safety mode and publish it back from panda
      if (safety_mismatch and self.sm.frame*DT_CTRL > 10.) or pandaState.safetyRxChecksInvalid or self.mismatch_counter >= 200:
        self.events.add(EventName.controlsMismatch)

      if log.PandaState.FaultType.relayMalfunction in pandaState.faults:
        self.events.add(EventName.relayMalfunction)

    # Handle HW and system malfunctions
    # Order is very intentional here. Be careful when modifying this.
    # All events here should at least have NO_ENTRY and SOFT_DISABLE.
    num_events = len(self.events)

    not_running = {p.name for p in self.sm['managerState'].processes if not p.running and p.shouldBeRunning}
    if self.sm.recv_frame['managerState'] and len(not_running):
      if not_running != self.not_running_prev:
        cloudlog.event("process_not_running", not_running=not_running, error=True)
      self.not_running_prev = not_running
    if self.sm.recv_frame['managerState'] and (not_running - self.ignored_processes):
      self.events.add(EventName.processNotRunning)
    else:
      if not SIMULATION and not self.rk.lagging:
        if not self.sm.all_alive(self.camera_packets):
          self.events.add(EventName.cameraMalfunction)
        elif not self.sm.all_freq_ok(self.camera_packets):
          self.events.add(EventName.cameraFrameRate)
    if not REPLAY and self.rk.lagging:
      self.events.add(EventName.selfdrivedLagging)
    if self.sm['radarState'].radarErrors.canError:
      self.events.add(EventName.canError)
    elif self.sm['radarState'].radarErrors.radarUnavailableTemporary:
      self.events.add(EventName.radarTempUnavailable)
    elif any(self.sm['radarState'].radarErrors.to_dict().values()):
      self.events.add(EventName.radarFault)
    if not self.sm.valid['pandaStates']:
      self.events.add(EventName.usbError)
    if CS.canTimeout:
      self.events.add(EventName.canBusMissing)
    elif not CS.canValid:
      self.events.add(EventName.canError)

    # generic catch-all. ideally, a more specific event should be added above instead
    has_disable_events = self.events.contains(ET.NO_ENTRY) and (self.events.contains(ET.SOFT_DISABLE) or self.events.contains(ET.IMMEDIATE_DISABLE))
    no_system_errors = (not has_disable_events) or (len(self.events) == num_events)
    if not self.sm.all_checks() and no_system_errors:
      if not self.sm.all_alive():
        self.events.add(EventName.commIssue)
      elif not self.sm.all_freq_ok():
        self.events.add(EventName.commIssueAvgFreq)
      else:
        self.events.add(EventName.commIssue)

      logs = {
        'invalid': sorted([s for s, valid in self.sm.valid.items() if not valid]),
        'not_alive': sorted([s for s, alive in self.sm.alive.items() if not alive]),
        'not_freq_ok': sorted([s for s, freq_ok in self.sm.freq_ok.items() if not freq_ok]),
      }

      previous_logs = self.logged_comm_issue
      if logs != previous_logs:
        cloudlog.event("commIssue", error=True, **logs)
      self.logged_comm_issue = {k: list(v) for k, v in logs.items()}

      if logs['not_alive']:
        reason = 'not_alive'
      elif logs['not_freq_ok']:
        reason = 'not_freq_ok'
      elif logs['invalid']:
        reason = 'invalid'
      else:
        reason = 'unknown'

      frame = int(self.sm.frame)
      recv_frame: Dict[str, int] = {}
      frame_age: Dict[str, Optional[int]] = {}
      for name, frame_id in self.sm.recv_frame.items():
        try:
          numeric_id = int(frame_id)
        except (TypeError, ValueError):
          numeric_id = -1
        recv_frame[name] = numeric_id
        frame_age[name] = frame - numeric_id if numeric_id >= 0 else None

      manager_processes: List[Dict[str, Any]] = []
      process_diagnostics: List[Dict[str, Any]] = []
      if self.sm.recv_frame['managerState']:
        for proc_state in self.sm['managerState'].processes:
          base = {
            'name': proc_state.name,
            'running': bool(proc_state.running),
            'shouldBeRunning': bool(proc_state.shouldBeRunning),
            'pid': int(proc_state.pid),
            'exitCode': int(proc_state.exitCode),
            'supportsStacktrace': proc_state.name in STACKTRACE_CAPABLE_PROCESSES_SET,
            'ignored': proc_state.name in self.ignored_processes,
          }
          manager_processes.append(base)
          process_diagnostics.append(_collect_process_diagnostics(proc_state))

      issue_context_changed = (
        not self.comm_issue_active or
        previous_logs is None or
        any(sorted(previous_logs.get(k, [])) != logs[k] for k in logs) or
        reason != self.last_comm_issue_reason
      )

      if issue_context_changed:
        self.stacktrace_request_context.clear()
        self.self_stacktrace_token = None

      self.comm_issue_active = True
      self.last_comm_issue_reason = reason

      stacktrace_requests = []
      if manager_processes:
        stacktrace_requests = self._request_comm_issue_stacktraces(manager_processes, reason, frame)

      self_stacktrace = ''.join(traceback.format_stack())
      if self.self_stacktrace_token is None or issue_context_changed:
        service_monitor.log_process_stacktrace(name="selfdrived", trigger=f"comm_issue:{reason}", stacktrace=self_stacktrace)
      self.self_stacktrace_token = (frame, reason)

      events_mapping = self.events.get_events_mapping()
      seen_event_names = set()
      event_details = []
      for event_raw in self.events.names:
        event_name = self.events.get_event_name(event_raw)
        if event_name in seen_event_names:
          continue
        seen_event_names.add(event_name)
        types = sorted(events_mapping.get(event_raw, {}).keys())
        event_details.append({'name': event_name, 'types': types})

      ratekeeper_state = {
        'frame': int(self.rk.frame),
        'lagging': bool(self.rk.lagging),
        'remaining': float(self.rk.remaining),
        'interval': float(self.rk._interval),
        'print_delay_threshold': self.rk._print_delay_threshold,
        'last_monitor_time': getattr(self.rk, '_last_monitor_time', None),
        'next_frame_time': getattr(self.rk, '_next_frame_time', None),
        'avg_dt_samples': self.rk.avg_dt.count,
      }
      try:
        ratekeeper_state['avg_dt'] = float(self.rk.avg_dt.get_average())
      except Exception:
        pass

      thread_dump = [{
        'name': thread.name,
        'ident': thread.ident,
        'daemon': thread.daemon,
        'native_id': getattr(thread, 'native_id', None),
        'alive': thread.is_alive(),
      } for thread in threading.enumerate()]

      monotonic_now = time.monotonic()
      submaster_details: Dict[str, Dict[str, Any]] = {}
      for service_name in sorted(self.sm.services):
        tracker = self.sm.freq_tracker.get(service_name)
        detail: Dict[str, Any] = {
          'seen': bool(self.sm.seen[service_name]),
          'recv_time': float(self.sm.recv_time[service_name]),
          'recv_frame': int(self.sm.recv_frame[service_name]),
          'logMonoTime': int(self.sm.logMonoTime[service_name]),
          'valid': bool(self.sm.valid[service_name]),
          'alive': bool(self.sm.alive[service_name]),
          'freq_ok': bool(self.sm.freq_ok[service_name]),
          'updated': bool(self.sm.updated[service_name]),
          'age_sec': None,
        }
        if self.sm.recv_time[service_name]:
          detail['age_sec'] = max(0.0, monotonic_now - self.sm.recv_time[service_name])
        if service_name in messaging.SERVICE_LIST:
          detail['expected_frequency_hz'] = float(messaging.SERVICE_LIST[service_name].frequency)
        if tracker is not None:
          detail['min_freq'] = getattr(tracker, 'min_freq', None)
          detail['max_freq'] = getattr(tracker, 'max_freq', None)
          detail['avg_dt_samples'] = tracker.avg_dt.count
          detail['recent_avg_dt_samples'] = tracker.recent_avg_dt.count
          if tracker.avg_dt.count:
            detail['avg_dt'] = float(tracker.avg_dt.get_average())
          if tracker.recent_avg_dt.count:
            detail['recent_avg_dt'] = float(tracker.recent_avg_dt.get_average())
        submaster_details[service_name] = detail

      submaster_meta = {
        'services': list(self.sm.services),
        'ignore_alive': list(self.sm.ignore_alive),
        'ignore_valid': list(self.sm.ignore_valid),
        'ignore_avg_freq': list(self.sm.ignore_average_freq),
        'non_polled_services': list(self.sm.non_polled_services),
        'update_freq': float(self.sm.update_freq),
        'frame': frame,
      }

      try:
        panda_summaries = [_summarize_panda_state(ps) for ps in self.sm['pandaStates']]
      except Exception:
        panda_summaries = []

      try:
        device_state_summary = {
          'thermalStatus': int(self.sm['deviceState'].thermalStatus),
          'fanSpeedPercentDesired': float(self.sm['deviceState'].fanSpeedPercentDesired),
          'fanSpeedRpm': int(self.sm['deviceState'].fanSpeedRpm),
          'freeSpacePercent': float(self.sm['deviceState'].freeSpacePercent),
          'memoryUsagePercent': float(self.sm['deviceState'].memoryUsagePercent),
          'cpuTempC': [float(temp) for temp in self.sm['deviceState'].cpuTempC],
          'gpuTempC': [float(temp) for temp in self.sm['deviceState'].gpuTempC],
          'ambientTempC': float(self.sm['deviceState'].ambientTempC),
          'storageHealthy': bool(self.sm['deviceState'].storageHealthy),
          'voltage': float(self.sm['deviceState'].voltage),
          'current': float(self.sm['deviceState'].current),
        }
      except Exception:
        device_state_summary = {}

      try:
        peripheral_state_summary = {
          'fanSpeedRpm': int(self.sm['peripheralState'].fanSpeedRpm),
          'pandaType': int(self.sm['peripheralState'].pandaType),
          'ignitionLine': bool(self.sm['peripheralState'].ignitionLine),
          'ignitionCan': bool(self.sm['peripheralState'].ignitionCan),
          'usbHubPresent': bool(self.sm['peripheralState'].usbHubPresent),
        }
      except Exception:
        peripheral_state_summary = {}

      try:
        controls_state_summary = {
          'enabled': bool(self.sm['controlsState'].enabled),
          'active': bool(self.sm['controlsState'].active),
          'curvature': float(self.sm['controlsState'].curvature),
          'steeringAngleDesiredDeg': float(self.sm['controlsState'].steeringAngleDesiredDeg),
          'steeringAngleDeg': float(self.sm['controlsState'].steeringAngleDeg),
          'longControlActive': bool(self.sm['controlsState'].longControlActive),
        }
      except Exception:
        controls_state_summary = {}

      try:
        car_state_summary = {
          'vEgo': float(CS.vEgo),
          'aEgo': float(CS.aEgo),
          'gearShifter': int(CS.gearShifter),
          'standstill': bool(CS.standstill),
          'gasPressed': bool(CS.gasPressed),
          'brakePressed': bool(CS.brakePressed),
          'steeringPressed': bool(CS.steeringPressed),
          'leftBlinker': bool(CS.leftBlinker),
          'rightBlinker': bool(CS.rightBlinker),
        }
      except Exception:
        car_state_summary = {}

      service_monitor.log_comm_issue(
        reason=reason,
        status=logs,
        frame=frame,
        recv_frame=recv_frame,
        frame_age=frame_age,
        valid={name: bool(valid) for name, valid in self.sm.valid.items()},
        alive={name: bool(alive) for name, alive in self.sm.alive.items()},
        freq_ok={name: bool(freq_ok) for name, freq_ok in self.sm.freq_ok.items()},
        updated={name: bool(updated) for name, updated in self.sm.updated.items()},
        manager_processes=manager_processes,
        events=event_details,
        not_running=sorted(not_running),
        ignored_processes=sorted(self.ignored_processes),
        ratekeeper=ratekeeper_state,
        extra={
          'started': bool(started),
          'rk_lagging': bool(self.rk.lagging),
          'camera_packets': list(self.camera_packets),
          'sensor_packets': list(self.sensor_packets),
          'gps_packets': list(self.gps_packets),
          'thread_dump': thread_dump,
          'submaster': submaster_details,
          'submaster_meta': submaster_meta,
          'process_diagnostics': process_diagnostics,
          'stacktrace_requests': stacktrace_requests,
          'selfdrived_stacktrace': self_stacktrace,
          'panda_states': panda_summaries,
          'device_state': device_state_summary,
          'peripheral_state': peripheral_state_summary,
          'controls_state': controls_state_summary,
          'car_state': car_state_summary,
          'monotonic_time': monotonic_now,
          'wall_time': time.time(),
          'comm_issue_context': {
            'reason': reason,
            'issue_context_changed': bool(issue_context_changed),
          },
        },
      )
    else:
      self.logged_comm_issue = None
      self.comm_issue_active = False
      self.stacktrace_request_context.clear()
      self.self_stacktrace_token = None
      self.last_comm_issue_reason = None

    if not self.CP.notCar:
      if not self.sm['livePose'].posenetOK:
        self.events.add(EventName.posenetInvalid)
      if not self.sm['livePose'].inputsOK:
        self.events.add(EventName.locationdTemporaryError)
      if not self.sm['liveParameters'].valid and cal_status == log.LiveCalibrationData.Status.calibrated and not TESTING_CLOSET and (not SIMULATION or REPLAY):
        self.events.add(EventName.paramsdTemporaryError)

    # conservative HW alert. if the data or frequency are off, locationd will throw an error
    if any((self.sm.frame - self.sm.recv_frame[s])*DT_CTRL > 10. for s in self.sensor_packets):
      self.events.add(EventName.sensorDataInvalid)

    if not REPLAY:
      # Check for mismatch between openpilot and car's PCM
      cruise_mismatch = CS.cruiseState.enabled and (not self.enabled or not self.CP.pcmCruise)
      self.cruise_mismatch_counter = self.cruise_mismatch_counter + 1 if cruise_mismatch else 0
      if self.cruise_mismatch_counter > int(6. / DT_CTRL):
        self.events.add(EventName.cruiseMismatch)

    # Send a "steering required alert" if saturation count has reached the limit
    if CS.steeringPressed:
      self.last_steering_pressed_frame = self.sm.frame
    recent_steer_pressed = (self.sm.frame - self.last_steering_pressed_frame)*DT_CTRL < 2.0
    controlstate = self.sm['controlsState']
    lac = getattr(controlstate.lateralControlState, controlstate.lateralControlState.which())
    if lac.active and not recent_steer_pressed and not self.CP.notCar:
      clipped_speed = max(CS.vEgo, 0.3)
      actual_lateral_accel = controlstate.curvature * (clipped_speed**2)
      desired_lateral_accel = self.sm['modelV2'].action.desiredCurvature * (clipped_speed**2)
      undershooting = abs(desired_lateral_accel) / abs(1e-3 + actual_lateral_accel) > 1.2
      turning = abs(desired_lateral_accel) > 1.0
      # TODO: lac.saturated includes speed and other checks, should be pulled out
      if undershooting and turning and lac.saturated:
        self.events.add(EventName.steerSaturated)

    # Check for FCW
    stock_long_is_braking = self.enabled and not self.CP.openpilotLongitudinalControl and CS.aEgo < -1.25
    model_fcw = self.sm['modelV2'].meta.hardBrakePredicted and not CS.brakePressed and not stock_long_is_braking
    planner_fcw = self.sm['longitudinalPlan'].fcw and self.enabled
    if (planner_fcw or model_fcw) and not self.CP.notCar:
      self.events.add(EventName.fcw)

    # GPS checks
    gps_ok = self.sm.recv_frame[self.gps_location_service] > 0 and (self.sm.frame - self.sm.recv_frame[self.gps_location_service]) * DT_CTRL < 2.0
    if not gps_ok and self.sm['livePose'].inputsOK and (self.distance_traveled > 1500):
      self.events.add(EventName.noGps)
    if gps_ok:
      self.distance_traveled = 0
    self.distance_traveled += abs(CS.vEgo) * DT_CTRL

    # TODO: fix simulator
    if not SIMULATION or REPLAY:
      if self.sm['modelV2'].frameDropPerc > 20:
        self.events.add(EventName.modeldLagging)

    # mute canBusMissing event if in Park, as it sometimes may trigger a false alarm with MADS in Paused state
    if CS.gearShifter == car.CarState.GearShifter.park and self.mads.enabled:
      self.events.remove(EventName.canBusMissing)

    CruiseHelper.update(self, CS, self.events_sp, self.experimental_mode)

    # decrement personality on distance button press
    if self.CP.openpilotLongitudinalControl:
      if any(not be.pressed and be.type == ButtonType.gapAdjustCruise for be in CS.buttonEvents):
        if not self.experimental_mode_switched:
          self.personality = (self.personality - 1) % 3
          self.params.put_nonblocking('LongitudinalPersonality', self.personality)
          self.events.add(EventName.personalityChanged)
        self.experimental_mode_switched = False

    self.icbm.run(CS, self.sm['carControl'], self.sm['longitudinalPlanSP'], self.is_metric)

  def _request_comm_issue_stacktraces(self, processes: List[Dict[str, Any]], reason: str, frame: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    signal_name = signal.Signals(signal.SIGUSR1).name

    for proc in processes:
      entry = {
        'name': proc['name'],
        'pid': proc['pid'],
        'running': proc['running'],
        'supportsStacktrace': proc['supportsStacktrace'],
      }

      if not proc['supportsStacktrace']:
        entry['status'] = 'unsupported'
        results.append(entry)
        continue

      if proc['pid'] <= 0:
        entry['status'] = 'missing_pid'
        results.append(entry)
        continue

      if not proc['running']:
        entry['status'] = 'not_running'
        results.append(entry)
        continue

      existing = self.stacktrace_request_context.get(proc['name'])
      if existing is not None:
        entry['status'] = 'already_requested'
        entry['requested_frame'] = existing[0]
        entry['requested_reason'] = existing[1]
        results.append(entry)
        continue

      try:
        os.kill(proc['pid'], signal.SIGUSR1)
      except ProcessLookupError:
        entry['status'] = 'process_lookup_error'
      except PermissionError:
        entry['status'] = 'permission_denied'
      except OSError as err:
        entry['status'] = f'error:{err.__class__.__name__}'
      else:
        entry['status'] = 'signaled'
        entry['signal'] = signal_name
        self.stacktrace_request_context[proc['name']] = (frame, reason)
        service_monitor.log_stacktrace_request(
          name=proc['name'],
          pid=proc['pid'],
          signal=signal_name,
          reason=f"comm_issue:{reason}",
        )

      results.append(entry)

    return results

  def data_sample(self):
    _car_state = messaging.recv_one(self.car_state_sock)
    CS = _car_state.carState if _car_state else self.CS_prev

    self.sm.update(0)

    if not self.initialized:
      all_valid = CS.canValid and self.sm.all_checks()
      timed_out = self.sm.frame * DT_CTRL > 6.
      if all_valid or timed_out or (SIMULATION and not REPLAY):
        available_streams = VisionIpcClient.available_streams("camerad", block=False)
        if VisionStreamType.VISION_STREAM_ROAD not in available_streams:
          self.sm.ignore_alive.append('roadCameraState')
          self.sm.ignore_valid.append('roadCameraState')
        if VisionStreamType.VISION_STREAM_WIDE_ROAD not in available_streams:
          self.sm.ignore_alive.append('wideRoadCameraState')
          self.sm.ignore_valid.append('wideRoadCameraState')

        if REPLAY and any(ps.controlsAllowed for ps in self.sm['pandaStates']):
          self.state_machine.state = State.enabled

        self.initialized = True
        cloudlog.event(
          "selfdrived.initialized",
          dt=self.sm.frame*DT_CTRL,
          timeout=timed_out,
          canValid=CS.canValid,
          invalid=[s for s, valid in self.sm.valid.items() if not valid],
          not_alive=[s for s, alive in self.sm.alive.items() if not alive],
          not_freq_ok=[s for s, freq_ok in self.sm.freq_ok.items() if not freq_ok],
          error=True,
        )

    # When the panda and selfdrived do not agree on controls_allowed
    # we want to disengage openpilot. However the status from the panda goes through
    # another socket other than the CAN messages and one can arrive earlier than the other.
    # Therefore we allow a mismatch for two samples, then we trigger the disengagement.
    if not self.enabled:
      self.mismatch_counter = 0

    # All pandas not in silent mode must have controlsAllowed when openpilot is enabled
    if self.enabled and any(not ps.controlsAllowed for ps in self.sm['pandaStates']
           if ps.safetyModel not in IGNORED_SAFETY_MODES):
      self.mismatch_counter += 1

    return CS

  def update_alerts(self, CS):
    clear_event_types = set()
    if ET.WARNING not in self.state_machine.current_alert_types:
      clear_event_types.add(ET.WARNING)
    if self.enabled:
      clear_event_types.add(ET.NO_ENTRY)

    pers = LONGITUDINAL_PERSONALITY_MAP[self.personality]
    callback_args = [self.CP, CS, self.sm, self.is_metric,
                     self.state_machine.soft_disable_timer, pers]

    alerts = self.events.create_alerts(self.state_machine.current_alert_types, callback_args)
    alerts_sp = self.events_sp.create_alerts(self.state_machine.current_alert_types, callback_args)

    self.AM.add_many(self.sm.frame, alerts + alerts_sp)
    self.AM.process_alerts(self.sm.frame, clear_event_types)

  def publish_selfdriveState(self, CS):
    # selfdriveState
    ss_msg = messaging.new_message('selfdriveState')
    ss_msg.valid = True
    ss = ss_msg.selfdriveState
    ss.enabled = self.enabled
    ss.active = self.active
    ss.state = self.state_machine.state
    ss.engageable = not self.events.contains(ET.NO_ENTRY)
    ss.experimentalMode = self.experimental_mode
    ss.personality = self.personality

    ss.alertText1 = self.AM.current_alert.alert_text_1
    ss.alertText2 = self.AM.current_alert.alert_text_2
    ss.alertSize = self.AM.current_alert.alert_size
    ss.alertStatus = self.AM.current_alert.alert_status
    ss.alertType = self.AM.current_alert.alert_type
    ss.alertSound = self.AM.current_alert.audible_alert
    ss.alertHudVisual = self.AM.current_alert.visual_alert

    self.pm.send('selfdriveState', ss_msg)

    # onroadEvents - logged every second or on change
    if (self.sm.frame % int(1. / DT_CTRL) == 0) or (self.events.names != self.events_prev):
      ce_send = messaging.new_message('onroadEvents', len(self.events))
      ce_send.valid = True
      ce_send.onroadEvents = self.events.to_msg()
      self.pm.send('onroadEvents', ce_send)
    self.events_prev = self.events.names.copy()

    # selfdriveStateSP
    ss_sp_msg = messaging.new_message('selfdriveStateSP')
    ss_sp_msg.valid = True
    ss_sp = ss_sp_msg.selfdriveStateSP
    mads = ss_sp.mads
    mads.state = self.mads.state_machine.state
    mads.enabled = self.mads.enabled
    mads.active = self.mads.active
    mads.available = self.mads.enabled_toggle

    icbm = ss_sp.intelligentCruiseButtonManagement
    icbm.state = self.icbm.state
    icbm.sendButton = self.icbm.cruise_button
    icbm.vTarget = self.icbm.v_target

    self.pm.send('selfdriveStateSP', ss_sp_msg)

    # onroadEventsSP - logged every second or on change
    if (self.sm.frame % int(1. / DT_CTRL) == 0) or (self.events_sp.names != self.events_sp_prev):
      ce_send_sp = messaging.new_message('onroadEventsSP')
      ce_send_sp.valid = True
      ce_send_sp.onroadEventsSP.events = self.events_sp.to_msg()
      self.pm.send('onroadEventsSP', ce_send_sp)
    self.events_sp_prev = self.events_sp.names.copy()

  def step(self):
    CS = self.data_sample()
    self.update_events(CS)
    if not self.CP.passive and self.initialized:
      self.enabled, self.active = self.state_machine.update(self.events)
    if not self.CP.notCar:
      self.mads.update(CS)
    self.update_alerts(CS)

    self.publish_selfdriveState(CS)

    self.CS_prev = CS

  def params_thread(self, evt):
    while not evt.is_set():
      self.is_metric = self.params.get_bool("IsMetric")
      self.is_ldw_enabled = self.params.get_bool("IsLdwEnabled")
      self.disengage_on_accelerator = self.params.get_bool("DisengageOnAccelerator")
      self.experimental_mode = self.params.get_bool("ExperimentalMode") and self.CP.openpilotLongitudinalControl
      self.personality = self.params.get("LongitudinalPersonality", return_default=True)

      self.mads.read_params()
      time.sleep(0.1)

  def run(self):
    e = threading.Event()
    t = threading.Thread(target=self.params_thread, args=(e, ))
    try:
      t.start()
      while True:
        self.step()
        self.rk.monitor_time()
    finally:
      e.set()
      t.join()


def main():
  config_realtime_process(4, Priority.CTRL_HIGH)
  s = SelfdriveD()
  s.run()

if __name__ == "__main__":
  main()
