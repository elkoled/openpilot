from dataclasses import dataclass, field
from enum import IntFlag

from opendbc.car.structs import CarParams
from opendbc.car import Bus, CarSpecs, DbcDict, PlatformConfig, Platforms
from opendbc.car.lateral import AngleSteeringLimitsVM
from opendbc.car.docs_definitions import CarDocs, CarHarness, CarParts
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, uds

Ecu = CarParams.Ecu


class CarControllerParams:
  STEER_STEP = 1

  ANGLE_LIMITS: AngleSteeringLimitsVM = AngleSteeringLimitsVM(
    390,  # deg
    # the EPS drops out from ~3.1 m/s^2
    MAX_LATERAL_ACCEL=2.5,  # m/s^2
    # the EPS limits the angle rate to ~100 deg/s at standstill
    MAX_ANGLE_RATE=0.5,  # deg/frame
  )
  STEER_DRIVER_ALLOWANCE = 0.5  # Driver intervention threshold, Nm

  ACC_STEP = 2  # radar messages are sent at 50 Hz
  ACCEL_MAX = 2.0  # m/s^2
  ACCEL_MIN = -3.5  # m/s^2

  # engine braking runs out around -0.3 m/s^2, the stock radar uses the friction brakes from there
  BRAKE_ON_ACCEL = -0.25  # m/s^2
  BRAKE_OFF_ACCEL = -0.1  # m/s^2

  # wheel torque for the pitch compensated accel, from 208 and Corsa F drives
  TORQUE_LOOKUP_BP = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]  # m/s^2
  TORQUE_LOOKUP_V = [-400, -140, 170, 375, 615, 865, 1100]  # Nm
  TORQUE_MAX = 1000  # Nm, highest seen from the stock radar
  TORQUE_MIN = -400  # Nm


class PSASafetyFlags(IntFlag):
  LONG_CONTROL = 1
  EPS_CAM_BUS = 2


@dataclass
class PSACarDocs(CarDocs):
  package: str = "Adaptive Cruise Control (ACC) & Lane Assist"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.psa_a]))


@dataclass
class PSAPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {
    Bus.pt: 'psa_aee2010_r3',
  })


class CAR(Platforms):
  PSA_PEUGEOT_208 = PSAPlatformConfig(
    [PSACarDocs("Peugeot 208 2019-25")],
    CarSpecs(wheelbase=2.54, steerRatio=17.6),
  )
  PSA_PEUGEOT_508 = PSAPlatformConfig(
    [PSACarDocs("Peugeot 508 2019-23")],
    CarSpecs(wheelbase=2.79, steerRatio=17.6),
  )


# ABS serial number
PSA_DIAG_REQ = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL, 0x01])
PSA_DIAG_RESP = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL + 0x40, 0x01])
PSA_SERIAL_REQ = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER, 0xF1, 0x8C])
PSA_SERIAL_RESP = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40, 0xF1, 0x8C])

FW_QUERY_CONFIG = FwQueryConfig(
  fw_version_regex=br"\d{18}[0-9A-Z]?",
  requests=[
    Request(
      [PSA_DIAG_REQ, PSA_SERIAL_REQ],
      [PSA_DIAG_RESP, PSA_SERIAL_RESP],
      rx_offset=-0x20,
      bus=0,
      obd_multiplexing=False,
    ),
  ],
)

DBC = CAR.create_dbc_map()
