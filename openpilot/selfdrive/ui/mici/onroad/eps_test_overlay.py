import json
import os
import time
from collections import deque

import pyray as rl
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

# local MEB EPS test status, written by the car controller (opendbc volkswagen power_test)
STATUS_FILE = "/dev/shm/meb_power_test.json"
STALE_S = 2.
READ_EVERY_S = 0.05
TRACE_LEN = 200  # 4 s at 50 Hz

WHITE = rl.Color(255, 255, 255, 255)
GREY = rl.Color(255, 255, 255, 140)
DIM = rl.Color(255, 255, 255, 60)
GREEN = rl.Color(51, 220, 120, 255)
YELLOW = rl.Color(255, 200, 40, 255)
RED = rl.Color(255, 70, 70, 255)
BLUE = rl.Color(90, 170, 255, 255)


class EpsTestOverlay(Widget):
  def __init__(self):
    super().__init__()
    self._status: dict | None = None
    self._read_t = 0.
    self._mtime = 0.
    self._trace: deque[tuple[float, float]] = deque(maxlen=TRACE_LEN)
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font = gui_app.font(FontWeight.MEDIUM)

  def visible(self) -> bool:
    self._read()
    return self._status is not None

  def _read(self):
    now = time.monotonic()
    if now - self._read_t < READ_EVERY_S:
      return
    self._read_t = now
    try:
      mtime = os.path.getmtime(STATUS_FILE)
      if mtime != self._mtime:
        with open(STATUS_FILE) as f:
          status = json.load(f)
        self._mtime = mtime
        self._status = status
        self._trace.extend(tuple(p) for p in status.get("trace", []))
    except (OSError, ValueError):
      self._status = None
    # the car controller writes it at 10 Hz on the same monotonic clock
    if self._status is not None and now - self._status["t"] > STALE_S:
      self._status = None

  def _text(self, text: str, x: float, y: float, size: int, color: rl.Color, bold: bool = False, right: bool = False) -> float:
    font = self._font_bold if bold else self._font
    width = measure_text_cached(font, text, size).x
    rl.draw_text_ex(font, text, rl.Vector2(x - width if right else x, y), size, 0, color)
    return width

  def _fit(self, text: str, size: int, width: float) -> int:
    while size > 20 and measure_text_cached(self._font_bold, text, size).x > width:
      size -= 2
    return size

  def _render(self, rect: rl.Rectangle):
    s = self._status
    if s is None:
      return
    rl.draw_rectangle_rec(rect, rl.BLACK)
    x0, x1 = rect.x + 16, rect.x + rect.width - 16

    fault_now = s["hca_status"] in ("FAULT", "REJECTED", "PREEMPTED", "DISABLED")
    if s["active"]:
      color = BLUE if s.get("recenter") else GREEN
      title = "back to line" if s.get("recenter") else s["maneuver"]
    else:
      color = RED if fault_now or s["note"].startswith("EPS") else YELLOW
      title = s["note"] or "waiting"

    # what is happening and the steering power
    power = s["power"]
    power_text = f"{power}%"
    power_w = self._text(power_text, x1, rect.y + 6, 44, YELLOW if power > 50 else WHITE, bold=True, right=True)
    power_w += self._text("power", x1 - power_w - 8, rect.y + 24, 18, GREY, bold=True, right=True) + 8
    self._text(title, x0, rect.y + 8, self._fit(title, 40, x1 - x0 - power_w - 20), color, bold=True)

    # command vs measured curvature
    self._plot(rl.Rectangle(x0, rect.y + 60, x1 - x0, 118))

    # speed against the band, bands done, EPS state
    y = rect.y + 186
    if s["band_speed"] is not None:
      v, c, tol = s["v_ego"], s["band_speed"], s["band_tol"]
      w = self._text(f"{v:.1f}", x0, y, 38, GREEN if abs(v - c) <= tol else YELLOW, bold=True)
      self._text(f"/ {c:.1f} m/s", x0 + w + 8, y + 12, 22, GREY, bold=True)
    for i, done in enumerate(s["bands_done"]):
      cx, cy = rect.x + rect.width / 2 - 30 + i * 20, y + 22
      if done:
        rl.draw_circle(int(cx), int(cy), 7, GREEN)
      else:
        rl.draw_circle_lines(int(cx), int(cy), 7, WHITE if i == s["band"] else DIM)
    caps = [c for c in s["level_cap"] + s["jump_cap"] if c is not None]
    if fault_now:
      eps_text, eps_color = "EPS FAULT", RED
    elif caps:
      eps_text, eps_color = f"faults >= {min(caps)}%", RED
    else:
      eps_text, eps_color = "EPS OK", GREEN
    self._text(eps_text, x1, y + 8, 26, eps_color, bold=True, right=True)

    # overall progress
    bar = rl.Rectangle(rect.x, rect.y + rect.height - 4, rect.width, 4)
    rl.draw_rectangle_rec(bar, DIM)
    rl.draw_rectangle_rec(rl.Rectangle(bar.x, bar.y, bar.width * s["progress"], bar.height), color)

  def _plot(self, r: rl.Rectangle):
    rl.draw_rectangle_rounded_lines_ex(r, 0.08, 6, 1, DIM)
    rl.draw_line_ex(rl.Vector2(r.x, r.y + r.height / 2), rl.Vector2(r.x + r.width, r.y + r.height / 2), 1, DIM)
    if len(self._trace) < 2:
      return
    scale = max(0.02, max(max(abs(a), abs(b)) for a, b in self._trace)) * 1.1
    n = len(self._trace)
    for idx, color in ((1, GREEN), (0, WHITE)):
      pts = [rl.Vector2(r.x + r.width * (i + TRACE_LEN - n) / (TRACE_LEN - 1), r.y + r.height / 2 * (1 - p[idx] / scale))
             for i, p in enumerate(self._trace)]
      for a, b in zip(pts, pts[1:], strict=False):
        rl.draw_line_ex(a, b, 3, color)
    rl.draw_rectangle_rec(rl.Rectangle(r.x + 2, r.y + 2, 92, 22), rl.BLACK)
    rl.draw_rectangle_rec(rl.Rectangle(r.x + r.width - 92, r.y + 2, 90, 22), rl.BLACK)
    self._text("cmd", r.x + 8, r.y + 4, 18, WHITE, bold=True)
    self._text("eps", r.x + 54, r.y + 4, 18, GREEN, bold=True)
    self._text(f"{scale * 2600:.0f} deg", r.x + r.width - 8, r.y + 4, 18, GREY, bold=True, right=True)
