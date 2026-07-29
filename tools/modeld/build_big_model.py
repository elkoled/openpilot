#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from openpilot.common.file_chunker import get_chunk_targets, chunk_file
from openpilot.selfdrive.modeld.big_model_artifact import (
  ARTIFACT_NAME,
  DESCRIPTOR_PATH,
  MANIFEST_NAME,
  get_artifact_dir,
  get_descriptor,
  get_manifest,
  sha256,
  verify_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
MODELD_DIR = ROOT / "openpilot/selfdrive/modeld"
MODELS_DIR = MODELD_DIR / "models"
BUILD_ENV = {
  "DEBUG": "2",
  "DEV": "USB+AMD:LLVM",
  "FLOAT16": "1",
  "GMMU": "0",
  "JIT_BATCH_SIZE": "0",
  "WARP_DEV": "QCOM",
}
BUILD_ARGS = [
  "--model-size", "512x256",
  "--camera-resolutions", "1928x1208", "1344x760",
  "--onnx", str(MODELS_DIR / "big_driving_supercombo.onnx"),
  "--frame-skip", "4",
]
INPUTS = [
  "openpilot/common/file_chunker.py",
  "openpilot/common/hardware/hw.py",
  "openpilot/common/transformations/camera.py",
  "openpilot/common/transformations/model.py",
  "openpilot/selfdrive/modeld/big_model_artifact.py",
  "openpilot/selfdrive/modeld/compile_modeld.py",
  "openpilot/selfdrive/modeld/constants.py",
  "openpilot/selfdrive/modeld/get_model_metadata.py",
  "openpilot/selfdrive/modeld/helpers.py",
  "openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx",
  "openpilot/system/camerad/cameras/nv12_info.py",
  "tinygrad_repo/tinygrad",
  "tools/modeld/build_big_model.py",
]


def update_hash(h, path: Path, name: str) -> None:
  h.update(name.encode())
  with open(path, "rb") as f:
    while chunk := f.read(1024 * 1024):
      h.update(chunk)


def fingerprint() -> str:
  h = hashlib.sha256(json.dumps([BUILD_ENV, BUILD_ARGS, f"{sys.version_info.major}.{sys.version_info.minor}"],
                                sort_keys=True).encode())
  for name in INPUTS:
    path = ROOT / name
    if path.is_dir():
      for f in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        update_hash(h, f, str(f.relative_to(ROOT)))
    else:
      update_hash(h, path, name)
  return h.hexdigest()


def write_descriptor() -> None:
  with open(DESCRIPTOR_PATH, "w") as f:
    json.dump({"artifact_id": fingerprint(), "artifact_name": ARTIFACT_NAME}, f, indent=2, sort_keys=True)
    f.write("\n")


def build(output_dir: Path) -> None:
  assert platform.machine() == "aarch64" and Path("/TICI").is_file()
  assert get_descriptor()["artifact_id"] == fingerprint()
  output_dir.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(dir=output_dir.parent) as tmp:
    output = Path(tmp) / ARTIFACT_NAME
    env = {**os.environ, **BUILD_ENV, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "tinygrad_repo")))}
    subprocess.run([sys.executable, str(MODELD_DIR / "compile_modeld.py"), *BUILD_ARGS, "--output", str(output)],
                   cwd=ROOT, env=env, check=True)
    size, digest = output.stat().st_size, sha256(output)
    targets = get_chunk_targets(output_dir / ARTIFACT_NAME, size)
    chunk_file(output, targets)
    files = [{"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)} for path in targets[1:]]
    with open(output_dir / MANIFEST_NAME, "w") as f:
      json.dump({"artifact_id": fingerprint(), "files": files, "sha256": digest, "size": size}, f, indent=2, sort_keys=True)
      f.write("\n")


def validate_package(directory: Path) -> None:
  manifest, files = get_manifest(directory / MANIFEST_NAME)
  artifact_hash = hashlib.sha256()
  for entry in files:
    path = directory / entry["name"]
    assert path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]
    with open(path, "rb") as f:
      while chunk := f.read(1024 * 1024):
        artifact_hash.update(chunk)
  assert artifact_hash.hexdigest() == manifest["sha256"]


def validate_runtime(directory: Path, camera_size: tuple[int, int], max_seconds: float) -> None:
  validate_package(directory)
  with tempfile.TemporaryDirectory(dir=directory.parent) as tmp:
    cache_root = Path(tmp)
    destination = get_artifact_dir(cache_root)
    destination.mkdir(parents=True)
    for path in directory.iterdir():
      if path.is_file():
        os.link(path, destination / path.name)
    assert verify_artifact(cache_root, check_hash=True)

    os.environ["BIG_MODEL_CACHE_ROOT"] = str(cache_root)
    os.environ["GMMU"] = "0"
    from openpilot.selfdrive.modeld.modeld import ModelState
    from tinygrad import Device

    start = time.monotonic()
    model = ModelState(*camera_size, usbgpu=True)
    model.warmup()
    Device.default.synchronize()
    elapsed = time.monotonic() - start
    del model
    gc.collect()
    assert elapsed < max_seconds

    with open(directory / MANIFEST_NAME) as f:
      manifest = json.load(f)
    manifest.setdefault("validation", {})[f"{camera_size[0]}x{camera_size[1]}"] = elapsed
    with open(directory / MANIFEST_NAME, "w") as f:
      json.dump(manifest, f, indent=2, sort_keys=True)
      f.write("\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("command", choices=("fingerprint", "write-descriptor", "build", "validate-package", "validate-runtime"))
  parser.add_argument("--directory", type=Path)
  parser.add_argument("--camera-resolution", default="1344x760")
  parser.add_argument("--max-seconds", type=float, default=60.)
  args = parser.parse_args()
  if args.command == "fingerprint":
    print(fingerprint())
  elif args.command == "write-descriptor":
    write_descriptor()
  elif args.command == "build":
    build(args.directory)
  elif args.command == "validate-package":
    validate_package(args.directory)
  elif args.command == "validate-runtime":
    validate_runtime(args.directory, tuple(map(int, args.camera_resolution.split("x"))), args.max_seconds)
