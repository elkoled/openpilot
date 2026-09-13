#!/usr/bin/env python3
"""Independent contract eval: checks a golden frame and safety invariants."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
required = {
  "watchdog": r"PINBALL_WATCHDOG_MS\s+200u",
  "command_id": r"PINBALL_COMMAND_ID\s+0x200u",
  "status_id": r"PINBALL_STATUS_ID\s+0x201u",
  "magic_0": r"PINBALL_PROTOCOL_MAGIC_0\s+0x50u",
  "magic_1": r"PINBALL_PROTOCOL_MAGIC_1\s+0x42u",
  "version": r"PINBALL_PROTOCOL_VERSION\s+1u",
  "status_watchdog": r"PINBALL_STATUS_WATCHDOG\s+\(1u << 7\)",
  "left_release": r"LEFT_RELEASE_US\s+1570u",
  "left_press": r"LEFT_PRESS_US\s+1770u",
  "right_release": r"RIGHT_RELEASE_US\s+1420u",
  "right_press": r"RIGHT_PRESS_US\s+1220u",
  "start_gpio": r"START_GPIO\s+4u",
  "start_direct": r"gpio_put\(START_GPIO, \(state & PINBALL_START_PRESSED\) != 0u\)",
  "pwm_period": r"PWM_PERIOD_US\s+3003u",
}
text = (ROOT / "protocol.h").read_text() + (ROOT / "main.c").read_text()
missing = [name for name, pattern in required.items() if not re.search(pattern, text)]
if missing:
  raise SystemExit(f"protocol eval: FAIL missing {missing}")
subprocess.run([str(ROOT / "tests" / "run.sh")], check=True)
print(f"protocol eval: PASS ({len(required)}/{len(required)} contract constants + behavioral suite)")
