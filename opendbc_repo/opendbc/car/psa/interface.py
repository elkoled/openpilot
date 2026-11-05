from opendbc.car import structs, get_safety_config
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.psa.carcontroller import CarController
from opendbc.car.psa.carstate import CarState
from opendbc.car.psa.values import CAR, PSASafetyFlags

TransmissionType = structs.CarParams.TransmissionType


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = 'psa'

    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.psa)]
    if candidate == CAR.PSA_PEUGEOT_508:
      ret.safetyConfigs[0].safetyParam |= PSASafetyFlags.EPS_CAM_BUS.value

    ret.dashcamOnly = False

    ret.steerActuatorDelay = 0.3
    ret.steerLimitTimer = 0.1
    ret.steerAtStandstill = True

    ret.steerControlType = structs.CarParams.SteerControlType.angle
    ret.radarUnavailable = True

    ret.alphaLongitudinalAvailable = True
    if alpha_long:
      ret.openpilotLongitudinalControl = True
      ret.safetyConfigs[0].safetyParam |= PSASafetyFlags.LONG_CONTROL.value

    # 0.3 s on torque, 0.4-0.7 s on the friction brakes
    ret.longitudinalActuatorDelay = 0.4
    # the integrator picks up drag and the launch, the torque map is speed independent
    ret.longitudinalTuning.kiBP = [0., 4., 13., 20.]
    ret.longitudinalTuning.kiV = [0.3, 0.4, 0.5, 0.8]

    return ret
