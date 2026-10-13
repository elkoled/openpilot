import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.shader_polygon import draw_polygon
from openpilot.system.ui.widgets import Widget

# a top down view of the model's lane lines and plan, smoothed over time
DISTANCE = 50.0  # meters the arrow spans
LATERAL_SCALE = 14.0  # px per meter, lateral is stretched and distance squished to fit the panel
BOTTOM_MARGIN = 36
TOP_MARGIN = 40
SMOOTH_RC = 0.25  # seconds
MATCH_DISTANCE = 1.5  # meters a lane line can move between frames and still be the same line

PATH_WIDTH = 10
LANE_WIDTH = 3
OUTER_LANE_WIDTH = 2
HEAD_LENGTH = 15  # length of each arrowhead arm
LEAD_BAR_WIDTH = 30

WHITE = rl.Color(255, 255, 255, 255)
GREEN = rl.Color(70, 230, 90, 255)
GREY = rl.Color(110, 110, 110, 255)
LIGHT_GREY = rl.Color(200, 200, 200, 255)
DIM = rl.Color(60, 60, 60, 255)
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


class _Line:
  def __init__(self, y: np.ndarray):
    self.y = y
    self.alpha = 0.4
    self.seen = True


class ArrowView(Widget):
  """The model's lane lines and planned path from above, with the car at the bottom center. Lines are tracked by position,
  so a line the car crosses during a lane change moves under it instead of jumping."""

  def __init__(self):
    super().__init__()
    self._k = (1 / gui_app.target_fps) / (SMOOTH_RC + 1 / gui_app.target_fps)
    self._xs = np.linspace(0.0, DISTANCE, 40)
    self._lines: list[_Line] = []
    self._plan: np.ndarray | None = None
    self._effort = 0.0
    self._curvature = 0.0
    self._lead = DISTANCE
    self._lead_alpha = 0.0

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
    for lane_line in model.laneLines:
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
    for line in self._lines:
      # solid while seen, fades in and out as lines appear and drop
      line.alpha = self._smooth(line.alpha, 1.0 if line.seen else 0.0)
    self._lines = [line for line in self._lines if line.seen or line.alpha > 0.02]

    # how much of the steering limit is used, torque output when the car has one
    lateral = sm['controlsState'].lateralControlState
    if lateral.which() == 'torqueState':
      effort = abs(lateral.torqueState.output)
    else:
      effort = abs(model.action.desiredCurvature) * sm['carState'].vEgo ** 2 / MAX_LAT_ACCEL
    self._effort = self._smooth(self._effort, min(effort, 1.0))
    self._curvature = self._smooth(self._curvature, model.action.desiredCurvature)

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


  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.BLACK)
    if not ui_state.sm.alive['modelV2'] or self._plan is None:
      return

    origin_x, origin_y = rect.x + rect.width / 2, rect.y + rect.height - BOTTOM_MARGIN
    long_scale = (rect.height - BOTTOM_MARGIN - TOP_MARGIN) / DISTANCE

    def to_screen(x, y):
      return origin_x + np.asarray(y) * LATERAL_SCALE, origin_y - np.asarray(x) * long_scale

    engaged = ui_state.status == UIStatus.ENGAGED
    override = ui_state.status == UIStatus.OVERRIDE
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))

    # lanes: the two lines around the car green engaged and white on override, the rest grey, all dim when off
    left = max((line for line in self._lines if line.y[0] < 0), key=lambda line: line.y[0], default=None)
    right = min((line for line in self._lines if line.y[0] >= 0), key=lambda line: line.y[0], default=None)
    # near the steering limit the outside line of the curve lights up, the car drifts out there on low torque
    # model frame, positive curvature turns right
    outside = left if self._curvature > 0 else right
    for line in self._lines:
      inner = line is left or line is right
      if engaged or override:
        color = (GREEN if engaged else WHITE) if inner else GREY
      else:
        color = DIM
      if line is outside and engaged:
        color = _effort_color(color, self._effort)
      color = rl.Color(color.r, color.g, color.b, int(255 * line.alpha))
      if color.a > 2:
        _strip(*to_screen(self._xs, line.y), LANE_WIDTH if inner else OUTER_LANE_WIDTH, color)

    # the planned path, ending in an arrowhead that flattens into a bar at the lead car, light grey on override, hidden when off
    if engaged or override:
      path_color = WHITE if engaged else LIGHT_GREY
      lead_t = self._lead_alpha
      end = DISTANCE + (self._lead - DISTANCE) * lead_t
      x = np.linspace(0.0, max(end, 4.0), 40)
      px, py = to_screen(x, np.interp(x, self._xs, self._plan))
      _strip(px, py, PATH_WIDTH, path_color)

      tip = rl.Vector2(float(px[-1]), float(py[-1]))
      dx, dy = px[-1] - px[-2], py[-1] - py[-2]
      norm = max(float(np.hypot(dx, dy)), 1e-6)
      dx, dy = dx / norm, dy / norm
      # arms swept back 45 degrees, opening to 90 into the lead bar
      angle = np.radians(45 + 45 * lead_t)
      length = HEAD_LENGTH + (LEAD_BAR_WIDTH / 2 - HEAD_LENGTH) * lead_t
      for side in (-1, 1):
        c, s = np.cos(angle), np.sin(angle) * side
        ax, ay = -(dx * c - dy * s), -(dx * s + dy * c)
        end_point = rl.Vector2(tip.x + ax * length, tip.y + ay * length)
        rl.draw_line_ex(tip, end_point, PATH_WIDTH, path_color)
        rl.draw_circle_v(end_point, PATH_WIDTH / 2, path_color)
      rl.draw_circle_v(tip, PATH_WIDTH / 2, path_color)

    rl.end_scissor_mode()
