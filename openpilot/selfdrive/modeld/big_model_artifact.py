#!/usr/bin/env python3
"""Build, publish, and install the precompiled USB-GPU driving model.

The large pickle is distributed as GitHub Release assets.  The small descriptor
in the source tree selects an artifact built from the exact model/compiler
inputs in this checkout.  updated installs it before an update is finalized, so
modeld never needs network access or a compiler at boot.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

# This script is invoked directly by updated and CI, including before the
# checkout has been installed as a Python package.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from openpilot.common.file_chunker import CHUNK_SIZE, get_chunk_name


SCHEMA_VERSION = 1
ARTIFACT_NAME = "big_driving_tinygrad.pkl"
REMOTE_MANIFEST_NAME = "big_model_manifest.json"
DESCRIPTOR_NAME = "big_model_artifact.json"
DEFAULT_CACHE_ROOT = Path(os.getenv("BIG_MODEL_CACHE_ROOT", "/data/model_cache/openpilot"))
MODELD_DIR = Path(__file__).resolve().parent
MODELS_DIR = MODELD_DIR / "models"
DESCRIPTOR_PATH = MODELS_DIR / DESCRIPTOR_NAME

# Any change to these inputs requires a newly compiled artifact. Keep this list
# deliberately broad: a stale pickle is much worse than an occasional rebuild.
FINGERPRINT_PATHS = (
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
)

BUILD_RECIPE = {
  "camera_resolutions": ["1928x1208", "1344x760"],
  "compile_environment": {
    "DEV": "USB+AMD:LLVM",
    "DEBUG": "2",
    "FLOAT16": "1",
    "GMMU": "0",
    "JIT_BATCH_SIZE": "0",
    "WARP_DEV": "QCOM",
  },
  "frame_skip": 4,
  "model_size": "512x256",
  "python": f"{sys.version_info.major}.{sys.version_info.minor}",
  "schema": SCHEMA_VERSION,
}


def _repo_root() -> Path:
  return REPO_ROOT


def _hash_file(h: Any, path: Path, relative_path: str) -> None:
  h.update(relative_path.encode())
  h.update(b"\0")
  with path.open("rb") as f:
    while chunk := f.read(4 * 1024 * 1024):
      h.update(chunk)
  h.update(b"\0")


def compatibility_id(root: Path | None = None) -> str:
  """Hash every input that can affect the serialized GPU program."""
  root = root or _repo_root()
  h = hashlib.sha256()
  h.update(json.dumps(BUILD_RECIPE, sort_keys=True, separators=(",", ":")).encode())
  h.update(b"\0")
  for relative in FINGERPRINT_PATHS:
    path = root / relative
    if path.is_dir():
      files = sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
      for f in files:
        _hash_file(h, f, f.relative_to(root).as_posix())
    else:
      _hash_file(h, path, relative)
  return h.hexdigest()


def load_descriptor(path: Path = DESCRIPTOR_PATH) -> dict[str, Any]:
  with path.open() as f:
    descriptor = json.load(f)
  if descriptor.get("schema") != SCHEMA_VERSION:
    raise ValueError(f"unsupported big model artifact schema in {path}")
  artifact_id = descriptor.get("artifact_id")
  if not isinstance(artifact_id, str) or len(artifact_id) != 64:
    raise ValueError(f"invalid artifact_id in {path}")
  return descriptor


def artifact_dir(descriptor: dict[str, Any] | None = None, cache_root: Path = DEFAULT_CACHE_ROOT) -> Path:
  descriptor = descriptor or load_descriptor()
  return cache_root / f"big-model-{descriptor['artifact_id']}"


def artifact_path(descriptor: dict[str, Any] | None = None, cache_root: Path = DEFAULT_CACHE_ROOT) -> Path:
  return artifact_dir(descriptor, cache_root) / ARTIFACT_NAME


def _sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    while chunk := f.read(4 * 1024 * 1024):
      h.update(chunk)
  return h.hexdigest()


def _expected_chunk_names(count: int) -> list[str]:
  if count <= 0 or count >= 100:
    raise ValueError(f"invalid chunk count: {count}")
  return [get_chunk_name(ARTIFACT_NAME, i, count) for i in range(count)]


def validate_remote_manifest(manifest: dict[str, Any], descriptor: dict[str, Any]) -> list[dict[str, Any]]:
  if manifest.get("schema") != SCHEMA_VERSION:
    raise ValueError("unsupported remote manifest schema")
  if manifest.get("artifact_id") != descriptor["artifact_id"]:
    raise ValueError("remote manifest is for a different compiler/model fingerprint")
  files = manifest.get("files")
  if not isinstance(files, list):
    raise ValueError("remote manifest has no file list")
  expected_names = _expected_chunk_names(manifest.get("chunk_count", 0))
  if [f.get("name") for f in files] != expected_names:
    raise ValueError("remote manifest chunk names are incomplete or out of order")
  for f in files:
    if not isinstance(f.get("size"), int) or not 0 < f["size"] <= CHUNK_SIZE:
      raise ValueError(f"invalid size for {f.get('name')}")
    if not isinstance(f.get("sha256"), str) or len(f["sha256"]) != 64:
      raise ValueError(f"invalid sha256 for {f.get('name')}")
  if not isinstance(manifest.get("sha256"), str) or len(manifest["sha256"]) != 64:
    raise ValueError("remote manifest has an invalid whole-artifact sha256")
  if sum(f["size"] for f in files) != manifest.get("total_size"):
    raise ValueError("remote manifest total size does not match its chunks")
  return files


def validate_installed(cache_root: Path = DEFAULT_CACHE_ROOT, descriptor_path: Path = DESCRIPTOR_PATH,
                       full_hash: bool = False) -> bool:
  """Cheap at boot; full_hash is used by updated and CI."""
  try:
    descriptor = load_descriptor(descriptor_path)
    directory = artifact_dir(descriptor, cache_root)
    with (directory / REMOTE_MANIFEST_NAME).open() as f:
      manifest = json.load(f)
    files = validate_remote_manifest(manifest, descriptor)
    count = len(files)
    if (directory / f"{ARTIFACT_NAME}.chunkmanifest").read_text().strip() != str(count):
      return False
    for entry in files:
      path = directory / entry["name"]
      if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
      if full_hash and _sha256(path) != entry["sha256"]:
        return False
    return True
  except (OSError, TypeError, ValueError, json.JSONDecodeError):
    return False


def _github_repository(root: Path) -> str:
  url = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=root, text=True).strip()
  # Supports https://github.com/owner/repo(.git) and git@github.com:owner/repo(.git).
  repo = url.split("github.com", 1)[-1].lstrip("/:").removesuffix(".git")
  if repo.count("/") != 1:
    raise ValueError(f"cannot derive GitHub repository from origin: {url}")
  return repo


def _download(url: str, destination: Path) -> None:
  request = urllib.request.Request(url, headers={"User-Agent": "openpilot-big-model-updater"})
  with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
    shutil.copyfileobj(response, out, length=4 * 1024 * 1024)


def usbgpu_present() -> bool:
  for device in Path("/sys/bus/usb/devices").glob("*"):
    try:
      if int((device / "idVendor").read_text(), 16) == 0xADD1 and \
         int((device / "idProduct").read_text(), 16) == 0x0001:
        return True
    except (OSError, ValueError):
      pass
  return False


def install(repository: str | None = None, root: Path | None = None, cache_root: Path = DEFAULT_CACHE_ROOT,
            descriptor_path: Path = DESCRIPTOR_PATH, base_url: str = "https://github.com") -> Path:
  root = root or _repo_root()
  descriptor = load_descriptor(descriptor_path)
  repository = repository or _github_repository(root)
  tag = f"big-model-{descriptor['artifact_id']}"
  release_url = f"{base_url.rstrip('/')}/{repository}/releases/download/{tag}"

  destination = artifact_dir(descriptor, cache_root)
  if validate_installed(cache_root, descriptor_path, full_hash=True):
    return destination

  cache_root.mkdir(parents=True, exist_ok=True)
  partial = cache_root / f".{tag}.partial"
  partial.mkdir(exist_ok=True)
  remote_manifest_path = partial / REMOTE_MANIFEST_NAME
  manifest_download = partial / f".{REMOTE_MANIFEST_NAME}.download"
  _download(f"{release_url}/{REMOTE_MANIFEST_NAME}", manifest_download)
  os.replace(manifest_download, remote_manifest_path)
  with remote_manifest_path.open() as f:
    manifest = json.load(f)
  files = validate_remote_manifest(manifest, descriptor)

  whole_hash = hashlib.sha256()
  for entry in files:
    path = partial / entry["name"]
    if not path.is_file() or path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
      download = partial / f".{entry['name']}.download"
      download.unlink(missing_ok=True)
      _download(f"{release_url}/{entry['name']}", download)
      if download.stat().st_size != entry["size"] or _sha256(download) != entry["sha256"]:
        download.unlink(missing_ok=True)
        raise ValueError(f"downloaded chunk failed verification: {entry['name']}")
      os.replace(download, path)
    with path.open("rb") as f:
      while chunk := f.read(4 * 1024 * 1024):
        whole_hash.update(chunk)
  if whole_hash.hexdigest() != manifest["sha256"]:
    raise ValueError("downloaded chunks do not reassemble to the published artifact")

  # The chunk manifest is the completion marker and is written last.
  (partial / f"{ARTIFACT_NAME}.chunkmanifest").write_text(str(len(files)))
  if destination.exists():
    shutil.rmtree(destination)
  os.replace(partial, destination)
  if not validate_installed(cache_root, descriptor_path, full_hash=True):
    raise ValueError("installed big model artifact failed final verification")
  return destination


def package(compiled: Path, output_dir: Path, descriptor_path: Path = DESCRIPTOR_PATH) -> dict[str, Any]:
  descriptor = load_descriptor(descriptor_path)
  if compatibility_id() != descriptor["artifact_id"]:
    raise ValueError("descriptor artifact_id is stale; regenerate it before building")
  output_dir.mkdir(parents=True, exist_ok=True)
  total_size = compiled.stat().st_size
  whole_sha256 = _sha256(compiled)
  count = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE
  names = _expected_chunk_names(count)
  files: list[dict[str, Any]] = []
  with compiled.open("rb") as src:
    for name in names:
      path = output_dir / name
      with path.open("wb") as dst:
        dst.write(src.read(CHUNK_SIZE))
      files.append({"name": name, "size": path.stat().st_size, "sha256": _sha256(path)})
  manifest = {
    "schema": SCHEMA_VERSION,
    "artifact_id": descriptor["artifact_id"],
    "artifact_name": ARTIFACT_NAME,
    "build_recipe": BUILD_RECIPE,
    "builder": {
      "machine": platform.machine(),
      "platform": platform.platform(),
      "python": platform.python_version(),
    },
    "chunk_count": count,
    "files": files,
    "sha256": whole_sha256,
    "total_size": total_size,
  }
  (output_dir / REMOTE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  return manifest


def validate_package(directory: Path, descriptor_path: Path = DESCRIPTOR_PATH) -> None:
  descriptor = load_descriptor(descriptor_path)
  with (directory / REMOTE_MANIFEST_NAME).open() as f:
    manifest = json.load(f)
  files = validate_remote_manifest(manifest, descriptor)
  h = hashlib.sha256()
  for entry in files:
    path = directory / entry["name"]
    if path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
      raise ValueError(f"packaged chunk failed verification: {entry['name']}")
    with path.open("rb") as f:
      while chunk := f.read(4 * 1024 * 1024):
        h.update(chunk)
  if h.hexdigest() != manifest.get("sha256"):
    raise ValueError("reassembled artifact hash does not match the compiled model")


def validate_load(directory: Path, descriptor_path: Path = DESCRIPTOR_PATH, max_seconds: float = 60.) -> float:
  """Load the packaged pickle on target hardware exactly as modeld will."""
  validate_package(directory, descriptor_path)
  with (directory / REMOTE_MANIFEST_NAME).open() as f:
    manifest = json.load(f)
  chunk_manifest = directory / f"{ARTIFACT_NAME}.chunkmanifest"
  chunk_manifest.write_text(str(manifest["chunk_count"]))
  try:
    os.environ["GMMU"] = "0"
    from openpilot.common.file_chunker import open_file_chunked
    from openpilot.selfdrive.modeld.helpers import load_oob
    start = time.monotonic()
    with open_file_chunked(directory / ARTIFACT_NAME) as stream:
      jits = load_oob(stream)
    elapsed = time.monotonic() - start
    expected = {"metadata", "run_policy", (1928, 1208), (1344, 760)}
    if set(jits) != expected:
      raise ValueError(f"loaded artifact has unexpected programs: {set(jits)!r}")
    if elapsed > max_seconds:
      raise TimeoutError(f"precompiled model load took {elapsed:.1f}s (limit {max_seconds:.1f}s)")
    manifest["validation"] = {"load_seconds": round(elapsed, 3), "max_seconds": max_seconds, "target": "USB+AMD:LLVM"}
    (directory / REMOTE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"precompiled big model loaded in {elapsed:.1f}s")
    return elapsed
  finally:
    chunk_manifest.unlink(missing_ok=True)


def compile_model(output: Path) -> None:
  if platform.machine() != "aarch64" or not Path("/TICI").is_file():
    raise RuntimeError("the production artifact must be compiled on an ARM64 comma device")
  env = {
    **os.environ,
    **BUILD_RECIPE["compile_environment"],
    "PYTHONPATH": os.pathsep.join((str(_repo_root()), str(_repo_root() / "tinygrad_repo"))),
  }
  cmd = [
    sys.executable, str(MODELD_DIR / "compile_modeld.py"),
    "--model-size", BUILD_RECIPE["model_size"],
    "--camera-resolutions", *BUILD_RECIPE["camera_resolutions"],
    "--onnx", str(MODELS_DIR / "big_driving_supercombo.onnx"),
    "--output", str(output),
    "--frame-skip", str(BUILD_RECIPE["frame_skip"]),
  ]
  subprocess.run(cmd, cwd=_repo_root(), env=env, check=True)


def _write_descriptor(path: Path) -> None:
  descriptor = {
    "schema": SCHEMA_VERSION,
    "artifact_id": compatibility_id(),
    "artifact_name": ARTIFACT_NAME,
  }
  path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command", required=True)
  subparsers.add_parser("fingerprint")
  write_descriptor = subparsers.add_parser("write-descriptor")
  write_descriptor.add_argument("--output", type=Path, default=DESCRIPTOR_PATH)
  build = subparsers.add_parser("build")
  build.add_argument("--output-dir", type=Path, required=True)
  install_parser = subparsers.add_parser("install")
  install_parser.add_argument("--repository")
  install_parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
  install_parser.add_argument("--descriptor", type=Path, default=DESCRIPTOR_PATH)
  install_parser.add_argument("--if-usbgpu", action="store_true")
  validate = subparsers.add_parser("validate")
  validate.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
  validate.add_argument("--descriptor", type=Path, default=DESCRIPTOR_PATH)
  validate.add_argument("--full-hash", action="store_true")
  validate_package_parser = subparsers.add_parser("validate-package")
  validate_package_parser.add_argument("--directory", type=Path, required=True)
  validate_package_parser.add_argument("--descriptor", type=Path, default=DESCRIPTOR_PATH)
  validate_load_parser = subparsers.add_parser("validate-load")
  validate_load_parser.add_argument("--directory", type=Path, required=True)
  validate_load_parser.add_argument("--descriptor", type=Path, default=DESCRIPTOR_PATH)
  validate_load_parser.add_argument("--max-seconds", type=float, default=60.)
  args = parser.parse_args()

  if args.command == "fingerprint":
    print(compatibility_id())
  elif args.command == "write-descriptor":
    _write_descriptor(args.output)
  elif args.command == "build":
    descriptor = load_descriptor()
    if compatibility_id() != descriptor["artifact_id"]:
      raise SystemExit("big_model_artifact.json is stale")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="big-model-build-", dir=args.output_dir.parent) as tmp:
      compiled = Path(tmp) / ARTIFACT_NAME
      compile_model(compiled)
      package(compiled, args.output_dir)
  elif args.command == "install":
    if not args.if_usbgpu or usbgpu_present():
      print(install(args.repository, cache_root=args.cache_root, descriptor_path=args.descriptor))
  elif args.command == "validate":
    if not validate_installed(args.cache_root, args.descriptor, args.full_hash):
      raise SystemExit("big model artifact validation failed")
  elif args.command == "validate-package":
    validate_package(args.directory, args.descriptor)
  elif args.command == "validate-load":
    validate_load(args.directory, args.descriptor, args.max_seconds)


if __name__ == "__main__":
  main()
