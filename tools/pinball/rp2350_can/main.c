#include <stdio.h>

#include "hardware/clocks.h"
#include "hardware/pwm.h"
#include "pico/stdlib.h"

#include "protocol.h"
#include "xl2515.h"

#define LEFT_GPIO 14u
#define RIGHT_GPIO 15u
#define START_GPIO 4u
#define LEFT_RELEASE_US 1570u
#define LEFT_PRESS_US 1770u
#define RIGHT_RELEASE_US 1420u
#define RIGHT_PRESS_US 1220u
#define PWM_PERIOD_US 3003u /* 333.0003 Hz */

static pinball_controller_t controller;

static void apply_outputs(uint8_t state) {
  pwm_set_gpio_level(LEFT_GPIO,
      (state & PINBALL_LEFT_PRESSED) ? LEFT_PRESS_US : LEFT_RELEASE_US);
  pwm_set_gpio_level(RIGHT_GPIO,
      (state & PINBALL_RIGHT_PRESSED) ? RIGHT_PRESS_US : RIGHT_RELEASE_US);
  gpio_put(START_GPIO, (state & PINBALL_START_PRESSED) != 0u);
}

static void init_servos(void) {
  gpio_set_function(LEFT_GPIO, GPIO_FUNC_PWM);
  gpio_set_function(RIGHT_GPIO, GPIO_FUNC_PWM);
  const uint slice = pwm_gpio_to_slice_num(LEFT_GPIO);
  pwm_config config = pwm_get_default_config();
  pwm_config_set_clkdiv(&config, (float)clock_get_hz(clk_sys) / 1000000.0f);
  pwm_config_set_wrap(&config, PWM_PERIOD_US - 1u);
  pwm_init(slice, &config, true);
  gpio_init(START_GPIO);
  gpio_set_dir(START_GPIO, GPIO_OUT);
  gpio_put(START_GPIO, 0);
  apply_outputs(0u);
}

static void send_status(void) {
  uint8_t status[PINBALL_FRAME_LEN];
  pinball_make_status(&controller, status);
  if (!xl2515_send(PINBALL_STATUS_ID, status, PINBALL_FRAME_LEN)) {
    controller.faults |= PINBALL_FAULT_CAN;
  } else {
    controller.faults &= (uint8_t)~PINBALL_FAULT_CAN;
  }
}

int main(void) {
  stdio_init_all();
  init_servos();
  if (!xl2515_init_500k()) {
    controller.faults |= PINBALL_FAULT_CAN;
    while (true) tight_loop_contents(); /* servos remain released */
  }

  uint32_t last_status_ms = 0u;
  while (true) {
    const uint32_t now_ms = to_ms_since_boot(get_absolute_time());
    uint16_t can_id;
    uint8_t data[8];
    uint8_t length;
    if (xl2515_receive(&can_id, data, &length)) {
      if (can_id == PINBALL_COMMAND_ID && length == PINBALL_FRAME_LEN &&
          pinball_apply_command(&controller, data, now_ms)) {
        apply_outputs(controller.state);
      } else if (can_id != PINBALL_COMMAND_ID || length != PINBALL_FRAME_LEN) {
        controller.faults |= PINBALL_FAULT_BAD_FRAME;
      }
      send_status();
      last_status_ms = now_ms;
    }

    if (pinball_watchdog_poll(&controller, now_ms)) {
      apply_outputs(0u);
      send_status();
      last_status_ms = now_ms;
    }
    if ((uint32_t)(now_ms - last_status_ms) >= 100u) {
      if (xl2515_error_flags() != 0u) controller.faults |= PINBALL_FAULT_CAN;
      send_status();
      last_status_ms = now_ms;
    }
    tight_loop_contents();
  }
}
