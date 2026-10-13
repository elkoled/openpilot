import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.shader_polygon import draw_polygon
from openpilot.system.ui.widgets import Widget

# a top down view of the model's lane lines and plan, smoothed over time
DISTANCE = 50.0  # meters the arrow spans
LATERAL_SCALE = 14.0  # px per meter, lateral is stretched and distance squished to fit the panel
BOTTOM_MARGIN = 0
TOP_MARGIN = 30
SMOOTH_RC = 0.25  # seconds
MATCH_DISTANCE = 1.5  # meters a lane line can move between frames and still be the same line

PATH_WIDTH = 10
LANE_WIDTH = 3
OUTER_LANE_WIDTH = 2
HEAD_LENGTH = 16
HEAD_WIDTH = 26
LEAD_BAR_WIDTH = 28
LEAD_BAR_HEIGHT = 7

WHITE = rl.Color(255, 255, 255, 255)
GREEN = rl.Color(70, 230, 90, 255)
GREY = rl.Color(110, 110, 110, 255)
ORANGE = rl.Color(255, 120, 30, 255)
RED = rl.Color(235, 45, 35, 255)
MAX_LAT_ACCEL = 3.0  # m/s^2, ISO limit openpilot steers to
HIGH_EFFORT = 0.65


def _blend(a: rl.Color, b: rl.Color, t: float) -> rl.Color:
  t = float(np.clip(t, 0, 1))
  return rl.Color(int(a.r + (b.r - a.r) * t), int(a.g + (b.g - a.g) * t), int(a.b + (b.b - a.b) * t), int(a.a + (b.a - a.a) * t))


def _effort_color(base: rl.Color, effort: float) -> rl.Color:
  # the lane line keeps its color until 65% of the steering limit, then orange, red at the limit
  if effort < HIGH_EFFORT:
    return _blend(base, ORANGE, (effort - HIGH_EFFORT + 0.1) / 0.1)
  return _blend(ORANGE, RED, (effort - HIGH_EFFORT) / (1 - HIGH_EFFORT))


def _strip(px: np.ndarray, py: np.ndarray, width: float, color: rl.Color):
  tx, ty = np.gradient(px), np.gradient(py)
  norm = np.maximum(np.hypot(tx, ty), 1e-6)
  nx, ny = -ty / norm * width / 2, tx / norm * width / 2
  # left edge first, then the right edge back, so the strip faces the camera
  points = np.concatenate([np.stack([px - nx, py - ny], axis=1), np.stack([px + nx, py + ny], axis=1)[::-1]]).astype(np.float32)
  lo, hi = points.min(axis=0), points.max(axis=0)
  draw_polygon(rl.Rectangle(lo[0], lo[1], hi[0] - lo[0] + 1, hi[1] - lo[1] + 1), points, color)


def _triangle(a: rl.Vector2, b: rl.Vector2, c: rl.Vector2, color: rl.Color):
  # raylib culls by winding, draw both
  rl.draw_triangle(a, b, c, color)
  rl.draw_triangle(a, c, b, color)


def _quad(center: rl.Vector2, dx: float, dy: float, half_width: float, half_height: float, color: rl.Color):
  hx, hy = -dy * half_width, dx * half_width
  tx, ty = dx * half_height, dy * half_height
  a, b = rl.Vector2(center.x - hx - tx, center.y - hy - ty), rl.Vector2(center.x + hx - tx, center.y + hy - ty)
  c, d = rl.Vector2(center.x + hx + tx, center.y + hy + ty), rl.Vector2(center.x - hx + tx, center.y - hy + ty)
  _triangle(a, b, c, color)
  _triangle(a, c, d, color)


class _Line:
  def __init__(self, y: np.ndarray):
    self.y = y
    self.alpha = 0.4
    self.prob = 0.0
    self.seen = True


class ArrowView(Widget):
  """The model's lane lines and planned path from above, with the car at the bottom center. Lines are tracked by position,
  so a line the car crosses during a lane change moves under it instead of jumping."""

  def __init__(self):
    super().__init__()
    self._k = (1 / gui_app.target_fps) / (SMOOTH_RC + 1 / gui_app.target_fps)
    self._xs = np.linspace(0.0, DISTANCE * 1.3, 40)
    self._lines: list[_Line] = []
    self._plan: np.ndarray | None = None
    self._effort = 0.0
    self._lead = DISTANCE
    self._lead_alpha = 0.0
    self._arrow_alpha = 0.0

  def _smooth(self, old, new):
    return old + (new - old) * self._k

  def _update_state(self):
    sm = ui_state.sm
    model = sm['modelV2']
    if len(model.position.x) < 2:
      return

    plan = np.interp(self._xs, model.position.x, model.position.y)
    self._plan = plan if self._plan is None else self._smooth(self._plan, plan)

    # match this frame's lines to the tracked ones by where they are next to the car
    for line in self._lines:
      line.seen = False
    probs = list(model.laneLineProbs)
    for i, lane_line in enumerate(model.laneLines):
      if len(lane_line.x) < 2:
        continue
      y = np.interp(self._xs, lane_line.x, lane_line.y)
      free = [t for t in self._lines if not t.seen]
      match = min(free, key=lambda t: abs(t.y[0] - y[0]), default=None)
      if match is not None and abs(match.y[0] - y[0]) < MATCH_DISTANCE:
        match.y = self._smooth(match.y, y)
      else:
        match = _Line(y)
        self._lines.append(match)
      match.seen = True
      match.prob = self._smooth(match.prob, probs[i] if i < len(probs) else 0.0)
    for line in self._lines:
      # confidence drops while changing lanes, keep the lines visible through it
      line.alpha = self._smooth(line.alpha, float(np.clip(line.prob * 2, 0.4, 1.0)) if line.seen else 0.0)
    self._lines = [line for line in self._lines if line.seen or line.alpha > 0.02]

    # how much of the steering limit is used, torque output when the car has one
    lateral = sm['controlsState'].lateralControlState
    if lateral.which() == 'torqueState':
      effort = abs(lateral.torqueState.output)
    else:
      effort = abs(model.action.desiredCurvature) * sm['carState'].vEgo ** 2 / MAX_LAT_ACCEL
    self._effort = self._smooth(self._effort, min(effort, 1.0))

    lead = None
    if sm.alive['radarState'] and sm['radarState'].leadOne.present:
      lead = sm['radarState'].leadOne.dRel
    elif len(model.leadsV3) and model.leadsV3[0].prob > 0.5:
      lead = model.leadsV3[0].x[0]
    if lead is not None and lead < DISTANCE:
      self._lead = self._smooth(self._lead, lead) if self._lead_alpha > 0.05 else lead
      self._lead_alpha = self._smooth(self._lead_alpha, 1.0)
    else:
      self._lead_alpha = self._smooth(self._lead_alpha, 0.0)

    self._arrow_alpha = self._smooth(self._arrow_alpha, 1.0 if ui_state.status == UIStatus.ENGAGED else 0.0)

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.BLACK)
    if not ui_state.sm.alive['modelV2'] or self._plan is None:
      return

    origin_x, origin_y = rect.x + rect.width / 2, rect.y + rect.height - BOTTOM_MARGIN
    long_scale = (rect.height - BOTTOM_MARGIN - TOP_MARGIN) / DISTANCE

    def to_screen(x, y):
      return origin_x + np.asarray(y) * LATERAL_SCALE, origin_y - np.asarray(x) * long_scale

    engaged = ui_state.status == UIStatus.ENGAGED
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))

    # lanes: the two lines around the car in the status color, the rest grey
    left = max((line for line in self._lines if line.y[0] < 0), key=lambda line: line.y[0], default=None)
    right = min((line for line in self._lines if line.y[0] >= 0), key=lambda line: line.y[0], default=None)
    # near the steering limit the line the car is closest to lights up
    near = min((line for line in (left, right) if line is not None), key=lambda line: abs(line.y[0]), default=None)
    for line in self._lines:
      inner = line is left or line is right
      color = GREEN if inner and engaged else GREY
      if line is near and engaged:
        color = _effort_color(color, self._effort)
      color = rl.Color(color.r, color.g, color.b, int(255 * line.alpha))
      if color.a > 2:
        _strip(*to_screen(self._xs, line.y), LANE_WIDTH if inner else OUTER_LANE_WIDTH, color)

    # the planned path, ending in an arrowhead or a bar at the lead car, only while engaged
    alpha = int(255 * self._arrow_alpha)
    if alpha > 2:
      path_color = rl.Color(255, 255, 255, alpha)
      lead_t = self._lead_alpha
      end = DISTANCE + (self._lead - DISTANCE) * lead_t
      head = HEAD_LENGTH / long_scale * (1 - lead_t)
      x = np.linspace(0.0, max(end - head, 4.0), 40)
      px, py = to_screen(x, np.interp(x, self._xs, self._plan))
      _strip(px, py, PATH_WIDTH, path_color)

      tip = rl.Vector2(float(px[-1]), float(py[-1]))
      dx, dy = px[-1] - px[-2], py[-1] - py[-2]
      norm = max(float(np.hypot(dx, dy)), 1e-6)
      dx, dy = dx / norm, dy / norm
      length = HEAD_LENGTH * (1 - lead_t)
      half_width = HEAD_WIDTH / 2 + (LEAD_BAR_WIDTH - HEAD_WIDTH) / 2 * lead_t
      if lead_t > 0.02:
        _quad(tip, dx, dy, half_width, LEAD_BAR_HEIGHT / 2 * lead_t, path_color)
      if length > 1:
        point = rl.Vector2(tip.x + dx * length, tip.y + dy * length)
        hx, hy = -dy * half_width, dx * half_width
        _triangle(point, rl.Vector2(tip.x - hx, tip.y - hy), rl.Vector2(tip.x + hx, tip.y + hy), path_color)

    rl.end_scissor_mode()
