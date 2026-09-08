#!/usr/bin/env python3
"""Compile the big driving model on the dedicated Chestnut CI bench."""
import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
MODEL = Path('/data/chestnut-ci-workspace/big_driving_tinygrad.pkl')
sys.path.insert(0, str(ROOT))


def preflight():
  if not Path('/AGNOS').is_file() or not Path('/data/disable_openpilot_autostart').is_file():
    raise RuntimeError('Requires an AGNOS bench with openpilot autostart disabled')
  onroad = Path('/data/params/d/IsOnroad')
  if onroad.exists() and onroad.read_bytes() != b'0':
    raise RuntimeError('Device reports onroad')
  processes = subprocess.check_output(['ps', '-eo', 'args='], text=True)
  if any(name in processes for name in ('manager.py', 'modeld.py', 'selfdrive.modeld.modeld', 'sunnypilot.modeld_v2.modeld')):
    raise RuntimeError('Stop the normal openpilot launcher on this dedicated bench before running CI')
  from openpilot.selfdrive.modeld.helpers import chestnut_present
  device_model = Path('/sys/firmware/devicetree/base/model').read_text().strip('\x00')
  if device_model.split('comma ')[-1] != 'mici' or not chestnut_present():
    raise RuntimeError('Requires MICI and a detected Chestnut with compatible firmware')
  if shutil.disk_usage(ROOT).free < 12 * 1024**3:
    raise RuntimeError('Requires at least 12 GiB free workspace storage')
  print(f'Bench: {platform.node()}, hardware: mici, Chestnut detected', flush=True)


def compile_model():
  preflight()
  from openpilot.common.transformations.camera import _os_fisheye
  from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE
  from openpilot.selfdrive.modeld.constants import ModelConstants
  MODEL.unlink(missing_ok=True)
  command = [sys.executable, str(ROOT / 'openpilot/selfdrive/modeld/compile_modeld.py'),
             '--model-size', 'x'.join(map(str, MEDMODEL_INPUT_SIZE)),
             '--camera-resolutions', f'{_os_fisheye.width}x{_os_fisheye.height}',
             '--frame-skip', str(ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ),
             '--onnx', str(ROOT / 'openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx'),
             '--output', str(MODEL)]
  # The compiler executes capture/replay and same/different-seed correctness checks.
  subprocess.run(['taskset', '-c', '7', *command], check=True, timeout=2400)
  if not MODEL.is_file() or MODEL.stat().st_size == 0:
    raise RuntimeError('Compiler did not produce a big-model artifact')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('phase', choices=['preflight', 'compile'])
  args = parser.parse_args()
  if args.phase == 'preflight':
    preflight()
  else:
    compile_model()
