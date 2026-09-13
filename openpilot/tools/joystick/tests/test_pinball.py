from openpilot.tools.joystick.joystickd import pinball_button_states


class CapnpLikeList:
  def __init__(self, values):
    self.values = values

  def __iter__(self):
    return iter(self.values)

  def __len__(self):
    return len(self.values)

  def __getitem__(self, index):
    if not isinstance(index, int):
      raise TypeError("an integer is required")
    return self.values[index]


def test_pinball_button_states_are_independent_and_direct():
  assert pinball_button_states([1, 0, 1]) == (1.0, 0.0, 1.0)
  assert pinball_button_states([0, 1]) == (0.0, 1.0, 0.0)
  assert pinball_button_states(CapnpLikeList([1, 0, 1])) == (1.0, 0.0, 1.0)
