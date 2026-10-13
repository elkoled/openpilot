import numpy as np
import pyray as rl
from openpilot.cereal import log

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.shader_polygon import draw_polygon
from openpilot.system.ui.widgets import Widget

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

# a stylized top down view of what the model is doing, not its raw plan
DISTANCE = 50.0  # meters the arrow spans
LANE = 3.7  # meters
LATERAL_SCALE = 14.0  # px per meter, lateral is stretched and distance squished to fit the panel
MAX_BEND = 2.4  # meters the arrow tip can bend sideways from curvature
BOTTOM_MARGIN = 0
TOP_MARGIN = 30

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


class ArrowView(Widget):
  """The road ahead as lanes and an arrow: the arrow sits where the car is in its lane, bends with the model's curvature,
  curves into the next lane on lane changes and stops at a lead car."""

  def __init__(self):
    super().__init__()
    dt = 1 / gui_app.target_fps
    self._bend = FirstOrderFilter(0.0, 0.4, dt)
    self._center = FirstOrderFilter(0.0, 0.3, dt)  # lane center relative to the car
    self._road = FirstOrderFilter(0.0, 0.6, dt)  # slides the road over once the car is in the new lane
    self._tip = FirstOrderFilter(0.0, 0.5, dt)  # arrow tip, relative to the current lane's center
    self._reveal = FirstOrderFilter(0.0, 0.3, dt)  # target lane line during a lane change
    self._effort = FirstOrderFilter(0.0, 0.2, dt)
    self._lead = FirstOrderFilter(DISTANCE, 0.3, dt)
    self._lead_alpha = FirstOrderFilter(0.0, 0.2, dt)
    self._direction = 0
    self._crossed = False

  def _update_state(self):
    sm = ui_state.sm
    model = sm['modelV2']

    # how much of the steering limit is used, torque output when the car has one
    lateral = sm['controlsState'].lateralControlState
    if lateral.which() == 'torqueState':
      effort = abs(lateral.torqueState.output)
    else:
      effort = abs(model.action.desiredCurvature) * sm['carState'].vEgo ** 2 / MAX_LAT_ACCEL
    self._effort.update(min(effort, 1.0))

    bend = model.action.desiredCurvature * DISTANCE ** 2 / 2
    self._bend.update(MAX_BEND * np.tanh(bend / MAX_BEND))

    meta = model.meta
    direction = {LaneChangeDirection.left: -1, LaneChangeDirection.right: 1}.get(meta.laneChangeDirection.raw, 0)
    if direction:
      self._direction = direction
    state = meta.laneChangeState
    changing = state == LaneChangeState.laneChangeStarting
    if not changing:
      self._crossed = False

    # where the car is in its lane, from the lane lines next to it
    lines, probs = model.laneLines, model.laneLineProbs
    if len(lines) > 2 and len(lines[1].y) and len(lines[2].y) and min(probs[1], probs[2]) > 0.3:
      center = float(np.clip((lines[1].y[0] + lines[2].y[0]) / 2, -LANE, LANE))
      jump = np.round((center - self._center.x) / LANE) * LANE
      if jump:
        # crossing into the next lane relabels the lines: same picture with the new lane as reference, then the road slides over
        self._center.x += jump
        self._road.x += jump
        self._tip.x -= jump
        self._crossed = changing
      self._center.update(center)
    self._road.update(0.0)

    # the arrow points to the lane center, leans on a pending lane change and curves into the target lane during one
    if changing and not self._crossed:
      tip = self._direction * LANE
    elif state == LaneChangeState.preLaneChange and direction:
      tip = direction * LANE * 0.2
    else:
      tip = 0.0
    self._tip.update(tip)
    self._reveal.update(1.0 if state in (LaneChangeState.preLaneChange, LaneChangeState.laneChangeStarting) and direction else 0.0)

    lead = None
    if sm.alive['radarState'] and sm['radarState'].leadOne.present:
      lead = sm['radarState'].leadOne.dRel
    elif len(model.leadsV3) and model.leadsV3[0].prob > 0.5:
      lead = model.leadsV3[0].x[0]
    if lead is not None and lead < DISTANCE:
      self._lead.update(lead)
      self._lead_alpha.update(1.0)
    else:
      self._lead_alpha.update(0.0)
      if self._lead_alpha.x < 0.05:
        self._lead.update(DISTANCE)

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.BLACK)
    if not ui_state.sm.alive['modelV2']:
      return

    origin_x, origin_y = rect.x + rect.width / 2, rect.y + rect.height - BOTTOM_MARGIN
    long_scale = (rect.height - BOTTOM_MARGIN - TOP_MARGIN) / DISTANCE

    def to_screen(x, y):
      # lateral is relative to the current lane's center, shifted while the road slides over after a lane change
      return origin_x + (y + self._road.x) * LATERAL_SCALE, origin_y - x * long_scale

    engaged = ui_state.status == UIStatus.ENGAGED
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))

    # lanes: the current one in the status color, its neighbors grey
    car = -self._center.x
    ls = np.linspace(0.0, DISTANCE + TOP_MARGIN / long_scale, 60)
    lane_curve = self._bend.x * (ls / DISTANCE) ** 2
    lane_color = GREEN if engaged else GREY
    # near the steering limit the line the car is pushed toward lights up
    if abs(car) > 0.15:
      near = -0.5 if car < 0 else 0.5
    else:
      near = -0.5 if self._bend.x > 0 else 0.5
    for offset in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5):
      inner = abs(offset) == 0.5
      color = lane_color if inner else GREY
      if offset == 1.5 * self._direction:
        color = _blend(GREY, lane_color, self._reveal.x)
      if offset == near and engaged:
        color = _effort_color(color, self._effort.x)
      _strip(*to_screen(ls, lane_curve + offset * LANE), LANE_WIDTH if inner else OUTER_LANE_WIDTH, color)

    path_color = WHITE if engaged else (GREY if ui_state.status == UIStatus.OVERRIDE else rl.Color(255, 255, 255, 90))

    # arrow from the car to where it is headed, shortened to the lead car
    lead_t = self._lead_alpha.x
    end = DISTANCE + (self._lead.x - DISTANCE) * lead_t
    head = HEAD_LENGTH / long_scale * (1 - lead_t)
    x = np.linspace(0.0, max(end - head, 4.0), 50)
    v = x / x[-1]
    ease = 3 * v ** 2 - 2 * v ** 3
    px, py = to_screen(x, self._bend.x * (x / DISTANCE) ** 2 + car + (self._tip.x - car) * ease)
    _strip(px, py, PATH_WIDTH, path_color)

    tip = rl.Vector2(float(px[-1]), float(py[-1]))
    dx, dy = px[-1] - px[-2], py[-1] - py[-2]
    norm = max(float(np.hypot(dx, dy)), 1e-6)
    dx, dy = dx / norm, dy / norm

    # the arrowhead flattens into a bar as a lead car appears
    length = HEAD_LENGTH * (1 - lead_t)
    half_width = HEAD_WIDTH / 2 + (LEAD_BAR_WIDTH - HEAD_WIDTH) / 2 * lead_t
    if lead_t > 0.02:
      _quad(tip, dx, dy, half_width, LEAD_BAR_HEIGHT / 2 * lead_t, path_color)
    if length > 1:
      point = rl.Vector2(tip.x + dx * length, tip.y + dy * length)
      hx, hy = -dy * half_width, dx * half_width
      _triangle(point, rl.Vector2(tip.x - hx, tip.y - hy), rl.Vector2(tip.x + hx, tip.y + hy), path_color)

    rl.end_scissor_mode()
