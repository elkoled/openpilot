#!/usr/bin/env python3
"""Bounded, synthetic Chestnut compile/inference check for upstream openpilot."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(os.environ.get('CHESTNUT_RESULTS', ROOT / 'results'))
MODEL = RESULTS / 'big_driving_tinygrad.pkl'
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
  from openpilot.selfdrive.modeld.constants import ModelConstants
  MODEL.unlink(missing_ok=True)
  command = [sys.executable, str(ROOT / 'openpilot/selfdrive/modeld/compile_modeld.py'),
             # MEDMODEL_INPUT_SIZE and _os_fisheye in this fork revision.
             '--model-size', '512x256', '--camera-resolutions', '1344x760',
             '--frame-skip', str(ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ),
             '--onnx', str(ROOT / 'openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx'),
             '--output', str(MODEL)]
  # The compiler executes capture/replay and same/different-seed correctness checks.
  with (RESULTS / 'compile.log').open('w') as output:
    subprocess.run(['taskset', '-c', '7', *command], check=True, timeout=2400, stdout=output, stderr=subprocess.STDOUT)
  if not MODEL.is_file() or MODEL.stat().st_size == 0:
    raise RuntimeError('Compiler did not produce a big-model artifact')


def smoke(runs):
  preflight()
  import numpy as np
  from openpilot.selfdrive.modeld.compile_modeld import MODELD_INPUTS, make_input_queues, nv12_copy_size
  from openpilot.selfdrive.modeld.constants import ModelConstants
  from openpilot.selfdrive.modeld.helpers import load_oob
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  from tinygrad.device import Device
  from tinygrad.helpers import DEV

  if Device.DEFAULT != 'AMD' or DEV.interface != 'USB':
    raise RuntimeError(f'Expected USB+AMD, got {DEV.value}')
  with MODEL.open('rb') as source:
    model = load_oob(source)
  if model['input_devices']['model'] != 'AMD':
    raise RuntimeError('Compiled model is not a Chestnut artifact')
  shapes = model['metadata']['input_shapes']
  resolution = (1344, 760)
  frame_size = nv12_copy_size(*get_nv12_info(*resolution)[:3])
  queues, npy, frames = make_input_queues(shapes, ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ,
                                        Device.DEFAULT, frame_size)
  rng = np.random.default_rng(42)
  timings = []
  for i in range(runs):
    for value in npy.values():
      value[:] = rng.standard_normal(value.shape).astype(value.dtype)
    for frame in frames.values():
      frame[:] = rng.integers(0, 256, size=frame.shape, dtype=np.uint8)
    Device.default.synchronize()
    start = time.monotonic()
    outputs = model['run_model'][resolution](**{k: queues[k] for k in MODELD_INPUTS})
    Device.default.synchronize()
    timings.append((time.monotonic() - start) * 1000)
    if not outputs or not all(np.isfinite(output.numpy()).all() for output in outputs):
      raise RuntimeError(f'Invalid model output at iteration {i + 1}')
    print(f'{i + 1}/{runs}: {timings[-1]:.2f} ms', flush=True)
  (RESULTS / 'inference.json').write_text(json.dumps({
    'runs': runs, 'device': Device.DEFAULT, 'mean_ms': float(np.mean(timings)), 'max_ms': max(timings),
  }, indent=2) + '\n')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('phase', choices=['preflight', 'compile', 'smoke'])
  parser.add_argument('--runs', type=int, choices=range(1, 1001), default=20, metavar='1..1000')
  args = parser.parse_args()
  os.chdir(ROOT)
  os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
  RESULTS.mkdir(exist_ok=True)
  start = time.monotonic()
  suite = ET.Element('testsuite', name='chestnut', tests='1', failures='0')
  case = ET.SubElement(suite, 'testcase', classname='chestnut', name=args.phase)
  try:
    if args.phase == 'preflight':
      preflight()
    elif args.phase == 'compile':
      compile_model()
    else:
      smoke(args.runs)
  except Exception:
    suite.set('failures', '1')
    ET.SubElement(case, 'failure').text = traceback.format_exc()
    raise
  finally:
    case.set('time', str(time.monotonic() - start))
    ET.ElementTree(suite).write(RESULTS / f'{args.phase}.xml', encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
  main()
