import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.shader_polygon import draw_polygon
from openpilot.system.ui.widgets import Widget

# birdseye squished to fit the panel: lateral meters are stretched, distance compressed
LATERAL_SCALE = 14.0  # px per meter
MAX_DISTANCE = 60.0  # meters shown ahead
BOTTOM_MARGIN = 14
TOP_MARGIN = 22

PATH_WIDTH = 7
EDGE_WIDTH = 2
LANE_WIDTH = 2
HEAD_LENGTH = 16
HEAD_WIDTH = 22
LEAD_BAR_WIDTH = 26
LEAD_BAR_HEIGHT = 6

WHITE = rl.Color(255, 255, 255, 255)
GREEN = rl.Color(70, 230, 90, 255)
GREY = rl.Color(170, 170, 170, 255)


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


class ArrowView(Widget):
  """Top down arrow of the model's path between its lane lines."""

  def _lead_distance(self) -> float | None:
    sm = ui_state.sm
    if sm.alive['radarState'] and sm['radarState'].leadOne.present:
      return float(sm['radarState'].leadOne.dRel)
    leads = sm['modelV2'].leadsV3
    if len(leads) and leads[0].prob > 0.5:
      return float(leads[0].x[0])
    return None

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle_rec(rect, rl.BLACK)
    model = ui_state.sm['modelV2']
    if not ui_state.sm.alive['modelV2'] or len(model.position.x) < 2:
      return

    ego_x, ego_y = rect.x + rect.width / 2, rect.y + rect.height - BOTTOM_MARGIN
    long_scale = (rect.height - BOTTOM_MARGIN - TOP_MARGIN) / MAX_DISTANCE

    def to_screen(x, y):
      return ego_x + np.asarray(y) * LATERAL_SCALE, ego_y - np.asarray(x) * long_scale

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))

    # lane lines, faded by confidence
    probs = list(model.laneLineProbs)
    for i, line in enumerate(model.laneLines):
      if len(line.x) < 2:
        continue
      s = np.arange(0.0, MAX_DISTANCE, 1.0)
      x = np.clip(s, line.x[0], line.x[-1])
      lx, ly = to_screen(x, np.interp(x, line.x, line.y))
      alpha = int(255 * np.clip(probs[i] if i < len(probs) else 0, 0, 1) * (1.0 if i in (1, 2) else 0.5))
      _strip(lx, ly, LANE_WIDTH, rl.Color(GREY.r, GREY.g, GREY.b, alpha))

    # path, resampled every meter, ending at the lead car if there is one
    xs, ys = np.array(model.position.x), np.array(model.position.y)
    lead = self._lead_distance()
    end = min(MAX_DISTANCE, xs[-1], lead if lead is not None else MAX_DISTANCE)
    head = 0.0 if lead is not None and lead < MAX_DISTANCE else HEAD_LENGTH / long_scale
    x = np.arange(0.0, max(end - head, 2.0), 0.5)
    px, py = to_screen(x, np.interp(x, xs, ys))

    if ui_state.status == UIStatus.ENGAGED:
      path_color, edge_color = WHITE, GREEN
    elif ui_state.status == UIStatus.OVERRIDE:
      path_color, edge_color = GREY, None
    else:
      path_color, edge_color = rl.Color(255, 255, 255, 110), None

    if edge_color is not None:
      _strip(px, py, PATH_WIDTH + 2 * EDGE_WIDTH, edge_color)
    _strip(px, py, PATH_WIDTH, path_color)

    tip = rl.Vector2(float(px[-1]), float(py[-1]))
    dx, dy = px[-1] - px[-3], py[-1] - py[-3]
    norm = max(float(np.hypot(dx, dy)), 1e-6)
    dx, dy = dx / norm, dy / norm
    if head == 0.0:
      # lead car: the arrowhead flattens into a bar at its distance
      hx, hy = -dy * LEAD_BAR_WIDTH / 2, dx * LEAD_BAR_WIDTH / 2
      tx, ty = dx * LEAD_BAR_HEIGHT / 2, dy * LEAD_BAR_HEIGHT / 2
      a, b = rl.Vector2(tip.x - hx - tx, tip.y - hy - ty), rl.Vector2(tip.x + hx - tx, tip.y + hy - ty)
      c, d = rl.Vector2(tip.x + hx + tx, tip.y + hy + ty), rl.Vector2(tip.x - hx + tx, tip.y - hy + ty)
      _triangle(a, b, c, path_color)
      _triangle(a, c, d, path_color)
    else:
      hx, hy = -dy * HEAD_WIDTH / 2, dx * HEAD_WIDTH / 2
      point = rl.Vector2(tip.x + dx * HEAD_LENGTH, tip.y + dy * HEAD_LENGTH)
      if edge_color is not None:
        e = EDGE_WIDTH
        _triangle(rl.Vector2(point.x + dx * e, point.y + dy * e), rl.Vector2(tip.x - hx * (1 + e / 8) - dx * e, tip.y - hy * (1 + e / 8) - dy * e),
                  rl.Vector2(tip.x + hx * (1 + e / 8) - dx * e, tip.y + hy * (1 + e / 8) - dy * e), edge_color)
      _triangle(point, rl.Vector2(tip.x - hx, tip.y - hy), rl.Vector2(tip.x + hx, tip.y + hy), path_color)

    rl.end_scissor_mode()
