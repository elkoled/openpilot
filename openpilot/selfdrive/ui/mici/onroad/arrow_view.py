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
BOTTOM_MARGIN = 14
TOP_MARGIN = 26

PATH_WIDTH = 7
EDGE_WIDTH = 2
LANE_WIDTH = 2
HEAD_LENGTH = 16
HEAD_WIDTH = 22
LEAD_BAR_WIDTH = 26
LEAD_BAR_HEIGHT = 6

WHITE = rl.Color(255, 255, 255, 255)
GREEN = rl.Color(70, 230, 90, 255)
GREY = rl.Color(150, 150, 150, 255)


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
  """The road ahead as lanes and an arrow: bends with the model's curvature, moves over on lane changes and stops at a lead car."""

  def __init__(self):
    super().__init__()
    dt = 1 / gui_app.target_fps
    self._bend = FirstOrderFilter(0.0, 0.4, dt)
    self._shift = FirstOrderFilter(0.0, 0.5, dt)  # arrow tip, relative to the lanes
    self._road = FirstOrderFilter(0.0, 0.5, dt)  # lanes, slide over once the car is in the new lane
    self._reveal = FirstOrderFilter(0.0, 0.3, dt)  # target lane line during a lane change
    self._direction = 0
    self._changing = False
    self._lead = FirstOrderFilter(DISTANCE, 0.3, dt)
    self._lead_alpha = FirstOrderFilter(0.0, 0.2, dt)
    self._lane_alpha = [FirstOrderFilter(0.0, 0.5, dt) for _ in range(4)]

  def _update_state(self):
    sm = ui_state.sm
    model = sm['modelV2']

    # sideways offset of the arrow tip from the model's curvature, softly limited
    bend = model.action.desiredCurvature * DISTANCE ** 2 / 2
    self._bend.update(MAX_BEND * np.tanh(bend / MAX_BEND))

    # lane change: the arrow leans, then curves into the target lane, then the lanes slide under it
    meta = model.meta
    direction = {LaneChangeDirection.left: -1, LaneChangeDirection.right: 1}.get(meta.laneChangeDirection.raw, 0)
    if direction:
      self._direction = direction
    state = meta.laneChangeState
    changing = state == LaneChangeState.laneChangeStarting
    if self._changing and not changing:
      # the target lane becomes the current one, same picture, new reference
      self._road.x += self._direction * LANE
    self._changing = changing
    if changing:
      shift = self._direction * LANE
    elif state == LaneChangeState.preLaneChange and direction:
      shift = direction * LANE * 0.2
    else:
      shift = 0.0
    self._shift.update(shift)
    self._road.update(0.0)
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

    probs = list(model.laneLineProbs) + [0.0] * 4
    for i, f in enumerate(self._lane_alpha):
      f.update(max(float(np.clip(probs[i], 0, 1)), 0.5) * (1.0 if i in (1, 2) else 0.4))

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.BLACK)
    if not ui_state.sm.alive['modelV2']:
      return

    ego_x, ego_y = rect.x + rect.width / 2, rect.y + rect.height - BOTTOM_MARGIN
    long_scale = (rect.height - BOTTOM_MARGIN - TOP_MARGIN) / DISTANCE
    s = np.linspace(0.0, DISTANCE, 60)
    u = s / DISTANCE
    curve = self._bend.x * u ** 2

    def to_screen(x, y):
      return ego_x + y * LATERAL_SCALE, ego_y - x * long_scale

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))

    # lanes
    lane_top = DISTANCE + TOP_MARGIN / long_scale
    ls = np.linspace(0.0, lane_top, 60)
    lane_curve = self._bend.x * (ls / DISTANCE) ** 2
    lane_alpha = {-2.5: self._lane_alpha[0].x * 0.5, -1.5: self._lane_alpha[0].x, -0.5: self._lane_alpha[1].x,
                  0.5: self._lane_alpha[2].x, 1.5: self._lane_alpha[3].x, 2.5: self._lane_alpha[3].x * 0.5}
    if self._direction:
      lane_alpha[1.5 * self._direction] = max(lane_alpha[1.5 * self._direction], self._reveal.x)
    for offset, a in lane_alpha.items():
      alpha = int(255 * min(a, 1.0))
      if alpha > 2:
        _strip(*to_screen(ls, lane_curve + offset * LANE + self._road.x), LANE_WIDTH, rl.Color(GREY.r, GREY.g, GREY.b, alpha))

    if ui_state.status == UIStatus.ENGAGED:
      path_color, edge_color = WHITE, GREEN
    elif ui_state.status == UIStatus.OVERRIDE:
      path_color, edge_color = GREY, None
    else:
      path_color, edge_color = rl.Color(255, 255, 255, 90), None

    # arrow, eased over into the next lane on lane changes, shortened to the lead car
    lead_t = self._lead_alpha.x
    end = DISTANCE + (self._lead.x - DISTANCE) * lead_t
    head = HEAD_LENGTH / long_scale * (1 - lead_t)
    keep = s <= max(end - head, 4.0)
    x = s[keep]
    v = x / max(x[-1], 1.0)  # ease over the drawn length so the tip always reaches the target lane
    shift = self._shift.x * (3 * v ** 2 - 2 * v ** 3)
    px, py = to_screen(x, curve[keep] + shift)

    if edge_color is not None:
      _strip(px, py, PATH_WIDTH + 2 * EDGE_WIDTH, edge_color)
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
      if edge_color is not None:
        e = EDGE_WIDTH
        grow = 1 + e / 6
        _triangle(rl.Vector2(point.x + dx * e * 1.5, point.y + dy * e * 1.5), rl.Vector2(tip.x - hx * grow - dx * e, tip.y - hy * grow - dy * e),
                  rl.Vector2(tip.x + hx * grow - dx * e, tip.y + hy * grow - dy * e), edge_color)
      _triangle(point, rl.Vector2(tip.x - hx, tip.y - hy), rl.Vector2(tip.x + hx, tip.y + hy), path_color)

    rl.end_scissor_mode()
