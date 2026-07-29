import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpilot.selfdrive.modeld import big_model_artifact as artifact


class TestBigModelArtifact(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.root = Path(self.tmp.name)
    self.artifact_id = "a" * 64
    self.descriptor = self.root / artifact.DESCRIPTOR_NAME
    self.descriptor.write_text(json.dumps({
      "schema": artifact.SCHEMA_VERSION,
      "artifact_id": self.artifact_id,
      "artifact_name": artifact.ARTIFACT_NAME,
    }))

  def tearDown(self):
    self.tmp.cleanup()

  def test_package_install_and_detect_corruption(self):
    compiled = self.root / artifact.ARTIFACT_NAME
    compiled.write_bytes(bytes(range(35)))
    release = self.root / "release"
    cache = self.root / "cache"

    with patch.object(artifact, "CHUNK_SIZE", 10), \
         patch.object(artifact, "compatibility_id", return_value=self.artifact_id):
      manifest = artifact.package(compiled, release, self.descriptor)

    def fake_download(url: str, destination: Path):
      shutil.copyfile(release / url.rsplit("/", 1)[-1], destination)

    with patch.object(artifact, "CHUNK_SIZE", 10), patch.object(artifact, "_download", side_effect=fake_download):
      destination = artifact.install("owner/repo", self.root, cache, self.descriptor)
      self.assertTrue(artifact.validate_installed(cache, self.descriptor, full_hash=True))
      self.assertEqual(manifest["chunk_count"], 4)

      local_cache = self.root / "local-cache"
      artifact._stage_local_package(release, local_cache, self.descriptor)
      self.assertTrue(artifact.validate_installed(local_cache, self.descriptor, full_hash=True))

      first_chunk = destination / manifest["files"][0]["name"]
      first_chunk.write_bytes(b"corrupt!!!")
      self.assertFalse(artifact.validate_installed(cache, self.descriptor, full_hash=True))

  def test_manifest_rejects_missing_or_reordered_chunks(self):
    descriptor = artifact.load_descriptor(self.descriptor)
    manifest = {
      "schema": artifact.SCHEMA_VERSION,
      "artifact_id": self.artifact_id,
      "chunk_count": 2,
      "total_size": 2,
      "files": [
        {"name": artifact.get_chunk_name(artifact.ARTIFACT_NAME, 1, 2), "size": 1, "sha256": "0" * 64},
        {"name": artifact.get_chunk_name(artifact.ARTIFACT_NAME, 0, 2), "size": 1, "sha256": "0" * 64},
      ],
    }
    with self.assertRaisesRegex(ValueError, "incomplete or out of order"):
      artifact.validate_remote_manifest(manifest, descriptor)

  def test_install_resumes_verified_chunks(self):
    compiled = self.root / artifact.ARTIFACT_NAME
    compiled.write_bytes(bytes(range(25)))
    release = self.root / "release"
    cache = self.root / "cache"
    with patch.object(artifact, "CHUNK_SIZE", 10), \
         patch.object(artifact, "compatibility_id", return_value=self.artifact_id):
      manifest = artifact.package(compiled, release, self.descriptor)

    calls: dict[str, int] = {}
    fail_name = manifest["files"][1]["name"]

    def flaky_download(url: str, destination: Path):
      name = url.rsplit("/", 1)[-1]
      calls[name] = calls.get(name, 0) + 1
      if name == fail_name and calls[name] == 1:
        raise OSError("interrupted")
      shutil.copyfile(release / name, destination)

    with patch.object(artifact, "CHUNK_SIZE", 10), patch.object(artifact, "_download", side_effect=flaky_download):
      with self.assertRaisesRegex(OSError, "interrupted"):
        artifact.install("owner/repo", self.root, cache, self.descriptor)
      artifact.install("owner/repo", self.root, cache, self.descriptor)

    self.assertEqual(calls[manifest["files"][0]["name"]], 1)
    self.assertEqual(calls[fail_name], 2)

  def test_artifact_path_is_versioned(self):
    descriptor = artifact.load_descriptor(self.descriptor)
    self.assertEqual(
      artifact.artifact_path(descriptor, self.root),
      self.root / f"big-model-{self.artifact_id}" / artifact.ARTIFACT_NAME,
    )


if __name__ == "__main__":
  unittest.main()
