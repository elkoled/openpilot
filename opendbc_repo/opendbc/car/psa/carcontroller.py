import math
import numpy as np
from opendbc.can.packer import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, structs, make_tester_present_msg
from opendbc.car.lateral import apply_steer_angle_limits_vm, get_max_angle_vm
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.psa import psacan
from opendbc.car.psa.psacan import AccMode
from opendbc.car.psa.values import CarControllerParams
from opendbc.car.vehicle_model import VehicleModel

LongCtrlState = structs.CarControl.Actuators.LongControlState

# pandad blocks the radar messages until it leaves ELM327 mode
RADAR_DISABLE_FRAME = 100
# the stock radar holds the car from HOLD_SPEED and launches until LAUNCH_SPEED
HOLD_SPEED = 0.3  # m/s
LAUNCH_SPEED = 1.0  # m/s


def get_safety_CP():
  # the 208 has the shortest wheelbase, so the lowest angle limits for every platform
  from opendbc.car.psa.interface import CarInterface
  return CarInterface.get_non_essential_params("PSA_PEUGEOT_208")


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.packer = CANPacker(dbc_names[Bus.main])
    self.VM = VehicleModel(get_safety_CP())
    self.apply_angle_last = 0
    self.lat_active = False
    self.status = 2
    self.acc_mode = AccMode.TORQUE

  def update(self, CC, CS, now_nanos):
    can_sends = []
    actuators = CC.actuators

    # lateral control
    if self.frame % CarControllerParams.STEER_STEP == 0:
      # don't take over a curve steered above the lateral accel limit, it can't be held
      max_angle = get_max_angle_vm(max(CS.out.vEgoRaw, 1), self.VM, CarControllerParams)
      self.lat_active = CC.latActive and (self.lat_active or abs(CS.out.steeringAngleDeg) < max_angle)

      apply_angle = apply_steer_angle_limits_vm(actuators.steeringAngleDeg, self.apply_angle_last, CS.out.vEgoRaw,
                                                CS.out.steeringAngleDeg, self.lat_active, CarControllerParams, self.VM)

      # EPS disengages on steering override, activation sequence 2->3->4 to re-engage
      # STATUS  -  0: UNAVAILABLE, 1: UNSELECTED, 2: READY, 3: AUTHORIZED, 4: ACTIVE
      if not self.lat_active:
        self.status = 2
      elif not CS.eps_active and not CS.out.steeringPressed:
        self.status = 2 if self.status == 4 else self.status + 1
      else:
        self.status = 4

      can_sends.append(psacan.create_lka_steering(self.packer, self.lat_active, apply_angle, self.status))

      self.apply_angle_last = apply_angle

    # longitudinal control
    if self.CP.openpilotLongitudinalControl:
      # keep the radar silent, two senders of the radar messages fault the ESP
      if self.frame >= RADAR_DISABLE_FRAME and self.frame % 100 == 0:
        can_sends.append(psacan.create_disable_radar() if CS.radar_alive else make_tester_present_msg(0x6b6, 1))

      if self.frame >= RADAR_DISABLE_FRAME and not CS.radar_alive and self.frame % CarControllerParams.ACC_STEP == 0:
        can_sends.extend(self.update_long(CC, CS))

    # resume the stock ACC every 3 s at standstill, it does not drive off after a longer stop
    elif CC.latActive and CS.out.standstill and CC.hudControl.leadVisible:
      status = {0: 0, 5: 1}.get(self.frame % 300)
      if status is not None:
        can_sends.append(psacan.create_resume_acc(self.packer, (CS.hs2_dat_mdd_cmd_452['COUNTER'] + 1) % 16, status, CS.hs2_dat_mdd_cmd_452))

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last
    self.frame += 1
    return new_actuators, can_sends

  def update_long(self, CC, CS):
    accel = float(np.clip(CC.actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))

    # the CMM takes a wheel torque, compensate it for the grade, the ESP brakes to the measured decel
    accel_pitch = accel
    if len(CC.orientationNED) == 3:
      accel_pitch += math.sin(CC.orientationNED[1]) * ACCELERATION_DUE_TO_GRAVITY
    torque = float(np.clip(np.interp(accel_pitch, CarControllerParams.TORQUE_LOOKUP_BP, CarControllerParams.TORQUE_LOOKUP_V),
                           CarControllerParams.TORQUE_MIN, CarControllerParams.TORQUE_MAX))

    stopping = CC.actuators.longControlState == LongCtrlState.stopping
    if not CC.longActive:
      self.acc_mode = AccMode.TORQUE
    elif stopping and CS.out.vEgo < HOLD_SPEED:
      self.acc_mode = AccMode.HOLD
    elif self.acc_mode == AccMode.HOLD:
      self.acc_mode = AccMode.LAUNCH
    elif accel_pitch < CarControllerParams.BRAKE_ON_ACCEL:
      # also leaves the launch, the lead can stop again before we are rolling
      self.acc_mode = AccMode.BRAKE
    elif self.acc_mode == AccMode.LAUNCH:
      if CS.out.vEgo >= LAUNCH_SPEED:
        self.acc_mode = AccMode.TORQUE
    elif accel_pitch > CarControllerParams.BRAKE_OFF_ACCEL:
      self.acc_mode = AccMode.TORQUE

    hud = CC.hudControl
    return [
      psacan.create_HS2_DYN1_MDD_ETAT_2B6(self.packer, CC.longActive, CC.cruiseControl.override, CS.out.brakePressed,
                                          self.acc_mode, accel, torque),
      psacan.create_HS2_DYN_MDD_ETAT_2F6(self.packer, CC.longActive and self.acc_mode != AccMode.TORQUE, hud.leadVisible,
                                         hud.leadDistanceBars),
    ]
