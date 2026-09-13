from openpilot.tools.joystick.joystickd import pinball_button_states


def test_pinball_button_states_are_independent_and_direct():
  assert pinball_button_states([1, 0, 1]) == (1.0, 0.0, 1.0)
  assert pinball_button_states([0, 1]) == (0.0, 1.0, 0.0)
