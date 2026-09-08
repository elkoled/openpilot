import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import run


class TestRunner(unittest.TestCase):
  def test_unmarked_host_rejected_before_hardware_imports(self):
    with patch.object(Path, 'is_file', return_value=False), self.assertRaisesRegex(RuntimeError, 'AGNOS bench'):
      run.preflight()

  def test_failure_propagates_and_writes_junit(self):
    with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'RESULTS', Path(tmp)), \
         patch.object(sys, 'argv', ['run.py', 'preflight']), \
         patch.object(run, 'preflight', side_effect=RuntimeError('GPU missing')):
      with self.assertRaisesRegex(RuntimeError, 'GPU missing'):
        run.main()
      suite = ET.parse(Path(tmp) / 'preflight.xml').getroot()
      self.assertEqual(suite.attrib['failures'], '1')
      self.assertIn('GPU missing', suite.find('testcase/failure').text)

  def test_success_writes_junit(self):
    with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'RESULTS', Path(tmp)), \
         patch.object(sys, 'argv', ['run.py', 'smoke', '--runs', '20']), patch.object(run, 'smoke') as smoke:
      run.main()
      smoke.assert_called_once_with(20)
      suite = ET.parse(Path(tmp) / 'smoke.xml').getroot()
      self.assertEqual(suite.attrib['failures'], '0')

  def test_out_of_bounds_runs_rejected(self):
    for count in ('0', '-1', '1001'):
      with self.subTest(count=count), patch.object(sys, 'argv', ['run.py', 'smoke', '--runs', count]), \
           contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
        run.main()
      self.assertEqual(error.exception.code, 2)

  def test_compile_timeout_is_a_failure(self):
    with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'RESULTS', Path(tmp)), \
         patch.object(sys, 'argv', ['run.py', 'compile']), \
         patch.object(run, 'compile_model', side_effect=subprocess.TimeoutExpired('compiler', 2400)):
      with self.assertRaises(subprocess.TimeoutExpired):
        run.main()
      suite = ET.parse(Path(tmp) / 'compile.xml').getroot()
      self.assertEqual(suite.attrib['failures'], '1')


if __name__ == '__main__':
  unittest.main()
