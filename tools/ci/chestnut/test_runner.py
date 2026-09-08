from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run


class TestRunner(unittest.TestCase):
  def test_unmarked_host_rejected_before_hardware_imports(self):
    with patch.object(Path, 'is_file', return_value=False), self.assertRaisesRegex(RuntimeError, 'AGNOS bench'):
      run.preflight()

  def test_onroad_host_rejected(self):
    with patch.object(Path, 'is_file', return_value=True), patch.object(Path, 'exists', return_value=True), \
         patch.object(Path, 'read_bytes', return_value=b'1'), self.assertRaisesRegex(RuntimeError, 'onroad'):
      run.preflight()

  def test_compile_failures_propagate(self):
    for error in (subprocess.CalledProcessError(1, 'compiler'), subprocess.TimeoutExpired('compiler', 2400)):
      with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp, \
           patch.object(run, 'MODEL', Path(tmp) / 'model.pkl'), patch.object(run, 'preflight'), \
           patch.object(run.subprocess, 'run', side_effect=error), self.assertRaises(type(error)):
        run.compile_model()

  def test_stale_artifact_cannot_pass(self):
    with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'MODEL', Path(tmp) / 'model.pkl'), \
         patch.object(run, 'preflight'), patch.object(run.subprocess, 'run'):
      run.MODEL.write_bytes(b'old artifact')
      with self.assertRaisesRegex(RuntimeError, 'did not produce'):
        run.compile_model()
      self.assertFalse(run.MODEL.exists())

  def test_empty_artifact_cannot_pass(self):
    with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'MODEL', Path(tmp) / 'model.pkl'), \
         patch.object(run, 'preflight'), patch.object(run.subprocess, 'run', side_effect=lambda *a, **kw: run.MODEL.touch()), \
         self.assertRaisesRegex(RuntimeError, 'did not produce'):
      run.compile_model()


if __name__ == '__main__':
  unittest.main()
