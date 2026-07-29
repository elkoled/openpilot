#!/usr/bin/env python3
import argparse
import statistics
import time

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--camera-resolution", default="1344x760")
  parser.add_argument("--runs", type=int, default=5)
  parser.add_argument("--max-ms", type=float, default=100.)
  args = parser.parse_args()

  from tinygrad import Device
  from openpilot.selfdrive.modeld.modeld import ModelState

  camera_size = tuple(map(int, args.camera_resolution.split("x")))
  model = ModelState(*camera_size, usbgpu=True)
  frames = {k: np.zeros(model.frame_buf_params[k][3], dtype=np.uint8) for k in model.vision_input_names}
  transforms = dict.fromkeys(model.vision_input_names, np.eye(3, dtype=np.float32))
  dims = {"desire_pulse": ModelConstants.DESIRE_LEN, "traffic_convention": 2, "action_t": 2}
  inputs = {k: np.zeros(v, dtype=np.float32) for k, v in dims.items()}

  model.warmup()
  runtimes = []
  for _ in range(args.runs):
    Device.default.synchronize()
    start = time.perf_counter()
    output = model.run(frames, transforms, inputs)
    Device.default.synchronize()
    runtimes.append((time.perf_counter() - start) * 1e3)
    assert output

  runtime_ms = statistics.median(runtimes)
  print(f"big model runtime: {runtime_ms:.2f} ms ({', '.join(f'{t:.2f}' for t in runtimes)})")
  assert runtime_ms <= args.max_ms, f"{runtime_ms:.2f} ms exceeds {args.max_ms:.2f} ms"
