import hashlib
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
    self.descriptor = self.root / "descriptor.json"
    self.descriptor.write_text(json.dumps({
      "artifact_id": self.artifact_id,
      "artifact_name": artifact.ARTIFACT_NAME,
    }))

  def tearDown(self):
    self.tmp.cleanup()

  def make_release(self, data: bytes) -> tuple[Path, dict]:
    release = self.root / "release"
    release.mkdir()
    count = (len(data) + artifact.CHUNK_SIZE - 1) // artifact.CHUNK_SIZE
    files = []
    for i in range(count):
      name = artifact.get_chunk_name(artifact.ARTIFACT_NAME, i, count)
      path = release / name
      path.write_bytes(data[i * artifact.CHUNK_SIZE:(i + 1) * artifact.CHUNK_SIZE])
      files.append({"name": name, "size": path.stat().st_size, "sha256": artifact.sha256(path)})
    manifest = {
      "artifact_id": self.artifact_id,
      "files": files,
      "sha256": hashlib.sha256(data).hexdigest(),
      "size": len(data),
    }
    (release / artifact.MANIFEST_NAME).write_text(json.dumps(manifest))
    return release, manifest

  def test_install_and_detect_corruption(self):
    cache = self.root / "cache"
    with patch.object(artifact, "CHUNK_SIZE", 10):
      release, manifest = self.make_release(bytes(range(35)))

      def fake_download(url: str, destination: Path):
        shutil.copyfile(release / url.rsplit("/", 1)[-1], destination)

      with patch.object(artifact, "download_file", side_effect=fake_download):
        destination = artifact.install(self.root, cache, self.descriptor, "owner/repo")

      self.assertTrue(artifact.verify_artifact(cache, self.descriptor, check_hash=True))
      first_chunk = destination / manifest["files"][0]["name"]
      first_chunk.write_bytes(b"corrupt!!!")
      self.assertFalse(artifact.verify_artifact(cache, self.descriptor, check_hash=True))

  def test_manifest_rejects_reordered_chunks(self):
    manifest = {
      "artifact_id": self.artifact_id,
      "files": [
        {"name": artifact.get_chunk_name(artifact.ARTIFACT_NAME, 1, 2), "size": 1, "sha256": "0" * 64},
        {"name": artifact.get_chunk_name(artifact.ARTIFACT_NAME, 0, 2), "size": 1, "sha256": "0" * 64},
      ],
      "sha256": "0" * 64,
      "size": 2,
    }
    path = self.root / artifact.MANIFEST_NAME
    path.write_text(json.dumps(manifest))
    with self.assertRaises(AssertionError):
      artifact.get_manifest(path, self.descriptor)

  def test_install_resumes_verified_chunks(self):
    cache = self.root / "cache"
    with patch.object(artifact, "CHUNK_SIZE", 10):
      release, manifest = self.make_release(bytes(range(25)))
      calls: dict[str, int] = {}
      fail_name = manifest["files"][1]["name"]

      def flaky_download(url: str, destination: Path):
        name = url.rsplit("/", 1)[-1]
        calls[name] = calls.get(name, 0) + 1
        if name == fail_name and calls[name] == 1:
          raise OSError("interrupted")
        shutil.copyfile(release / name, destination)

      with patch.object(artifact, "download_file", side_effect=flaky_download):
        with self.assertRaisesRegex(OSError, "interrupted"):
          artifact.install(self.root, cache, self.descriptor, "owner/repo")
        artifact.install(self.root, cache, self.descriptor, "owner/repo")

    self.assertEqual(calls[manifest["files"][0]["name"]], 1)
    self.assertEqual(calls[fail_name], 2)

  def test_artifact_path_is_versioned(self):
    self.assertEqual(
      artifact.get_artifact_path(self.root, self.descriptor),
      self.root / f"big-model-{self.artifact_id}" / artifact.ARTIFACT_NAME,
    )


if __name__ == "__main__":
  unittest.main()
