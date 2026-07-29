#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import requests

from openpilot.common.file_chunker import CHUNK_SIZE, get_chunk_name


ARTIFACT_NAME = "big_driving_tinygrad.pkl"
MANIFEST_NAME = "big_model_manifest.json"
MODELS_DIR = Path(__file__).resolve().parent / "models"
DESCRIPTOR_PATH = MODELS_DIR / "big_model_artifact.json"
DEFAULT_CACHE_ROOT = Path("/data/model_cache/openpilot")
USBGPU_VID = 0xADD1
USBGPU_PID = 0x0001


def get_descriptor(path: Path = DESCRIPTOR_PATH) -> dict:
  with open(path) as f:
    return json.load(f)


def get_cache_root() -> Path:
  return Path(os.getenv("BIG_MODEL_CACHE_ROOT", DEFAULT_CACHE_ROOT))


def get_artifact_dir(cache_root: Path | None = None, descriptor_path: Path = DESCRIPTOR_PATH) -> Path:
  cache_root = cache_root or get_cache_root()
  artifact_id = get_descriptor(descriptor_path)["artifact_id"]
  return cache_root / f"big-model-{artifact_id}"


def get_artifact_path(cache_root: Path | None = None, descriptor_path: Path = DESCRIPTOR_PATH) -> Path:
  return get_artifact_dir(cache_root, descriptor_path) / ARTIFACT_NAME


def sha256(path: Path) -> str:
  with open(path, "rb") as f:
    return hashlib.file_digest(f, "sha256").hexdigest()


def get_manifest(path: Path, descriptor_path: Path = DESCRIPTOR_PATH) -> tuple[dict, list[dict]]:
  with open(path) as f:
    manifest = json.load(f)
  descriptor = get_descriptor(descriptor_path)
  files = manifest["files"]
  assert manifest["artifact_id"] == descriptor["artifact_id"]
  assert [f["name"] for f in files] == [get_chunk_name(ARTIFACT_NAME, i, len(files)) for i in range(len(files))]
  assert all(0 < f["size"] <= CHUNK_SIZE and len(f["sha256"]) == 64 for f in files)
  assert sum(f["size"] for f in files) == manifest["size"]
  assert len(manifest["sha256"]) == 64
  return manifest, files


def verify_artifact(cache_root: Path | None = None, descriptor_path: Path = DESCRIPTOR_PATH, check_hash: bool = False) -> bool:
  try:
    directory = get_artifact_dir(cache_root, descriptor_path)
    manifest, files = get_manifest(directory / MANIFEST_NAME, descriptor_path)
    if (directory / f"{ARTIFACT_NAME}.chunkmanifest").read_text().strip() != str(len(files)):
      return False
    for entry in files:
      path = directory / entry["name"]
      if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
      if check_hash and sha256(path) != entry["sha256"]:
        return False
    return True
  except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
    return False


def get_github_repo(root: Path) -> str:
  url = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=root, text=True).strip()
  repo = url.split("github.com", 1)[-1].lstrip("/:").removesuffix(".git")
  assert repo.count("/") == 1
  return repo


def download_file(url: str, path: Path) -> None:
  with requests.get(url, stream=True, timeout=60) as response:
    response.raise_for_status()
    with open(path, "wb") as f:
      for chunk in response.iter_content(chunk_size=1024 * 1024):
        f.write(chunk)


def usbgpu_present() -> bool:
  for device in Path("/sys/bus/usb/devices").glob("*"):
    try:
      if int((device / "idVendor").read_text(), 16) == USBGPU_VID and \
         int((device / "idProduct").read_text(), 16) == USBGPU_PID:
        return True
    except (OSError, ValueError):
      pass
  return False


def install(root: Path, cache_root: Path | None = None, descriptor_path: Path = DESCRIPTOR_PATH,
            repository: str | None = None) -> Path:
  cache_root = cache_root or get_cache_root()
  descriptor = get_descriptor(descriptor_path)
  artifact_id = descriptor["artifact_id"]
  repository = repository or get_github_repo(root)
  url = f"https://github.com/{repository}/releases/download/big-model-{artifact_id}"
  destination = get_artifact_dir(cache_root, descriptor_path)

  if verify_artifact(cache_root, descriptor_path, check_hash=True):
    return destination

  cache_root.mkdir(parents=True, exist_ok=True)
  partial = cache_root / f".big-model-{artifact_id}.partial"
  partial.mkdir(exist_ok=True)
  download_file(f"{url}/{MANIFEST_NAME}", partial / MANIFEST_NAME)
  manifest, files = get_manifest(partial / MANIFEST_NAME, descriptor_path)

  artifact_hash = hashlib.sha256()
  for entry in files:
    path = partial / entry["name"]
    if not path.is_file() or path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
      download = path.with_suffix(path.suffix + ".download")
      download.unlink(missing_ok=True)
      download_file(f"{url}/{entry['name']}", download)
      assert download.stat().st_size == entry["size"] and sha256(download) == entry["sha256"]
      os.replace(download, path)
    with open(path, "rb") as f:
      while chunk := f.read(1024 * 1024):
        artifact_hash.update(chunk)

  assert artifact_hash.hexdigest() == manifest["sha256"]
  (partial / f"{ARTIFACT_NAME}.chunkmanifest").write_text(str(len(files)))
  if destination.exists():
    shutil.rmtree(destination)
  os.replace(partial, destination)
  assert verify_artifact(cache_root, descriptor_path, check_hash=True)
  return destination


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--if-usbgpu", action="store_true")
  args = parser.parse_args()
  if not args.if_usbgpu or usbgpu_present():
    install(args.root)
