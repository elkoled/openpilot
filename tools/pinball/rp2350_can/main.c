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
#define START_RELEASE_US 1420u
#define START_PRESS_US 1220u
#define PWM_PERIOD_US 3003u /* 333.0003 Hz */

static pinball_controller_t controller;

static void log_frame(uint16_t can_id, const uint8_t *data, uint8_t length, bool accepted) {
  printf("RX id=0x%03x dlc=%u data=", can_id, length);
  for (uint8_t i = 0; i < length; ++i) printf("%02x", data[i]);
  printf(" accepted=%u state=%u faults=0x%02x\n",
      accepted ? 1u : 0u, controller.state, controller.faults);
}

static void apply_outputs(uint8_t state) {
  pwm_set_gpio_level(LEFT_GPIO,
      (state & PINBALL_LEFT_PRESSED) ? LEFT_PRESS_US : LEFT_RELEASE_US);
  pwm_set_gpio_level(RIGHT_GPIO,
      (state & PINBALL_RIGHT_PRESSED) ? RIGHT_PRESS_US : RIGHT_RELEASE_US);
  pwm_set_gpio_level(START_GPIO,
      (state & PINBALL_START_PRESSED) ? START_PRESS_US : START_RELEASE_US);
}

static void init_servos(void) {
  gpio_set_function(LEFT_GPIO, GPIO_FUNC_PWM);
  gpio_set_function(RIGHT_GPIO, GPIO_FUNC_PWM);
  gpio_set_function(START_GPIO, GPIO_FUNC_PWM);
  pwm_config config = pwm_get_default_config();
  pwm_config_set_clkdiv(&config, (float)clock_get_hz(clk_sys) / 1000000.0f);
  pwm_config_set_wrap(&config, PWM_PERIOD_US - 1u);
  pwm_init(pwm_gpio_to_slice_num(LEFT_GPIO), &config, true);
  pwm_init(pwm_gpio_to_slice_num(START_GPIO), &config, true);
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
    while (true) {
      printf("CAN_INIT_FAILED faults=0x%02x\n", controller.faults);
      sleep_ms(1000); /* servos remain released */
    }
  }
  printf("READY can=500000 command=0x200 status=0x201 state=0\n");

  uint32_t last_status_ms = 0u;
  uint32_t last_log_ms = 0u;
  while (true) {
    const uint32_t now_ms = to_ms_since_boot(get_absolute_time());
    uint16_t can_id;
    uint8_t data[8];
    uint8_t length;
    if (xl2515_receive(&can_id, data, &length)) {
      const bool accepted = can_id == PINBALL_COMMAND_ID &&
                            ((length == 1u && pinball_apply_direct_state(&controller, data[0], now_ms)) ||
                             (length == PINBALL_FRAME_LEN && pinball_apply_command(&controller, data, now_ms)));
      if (accepted) {
        apply_outputs(controller.state);
      } else {
        controller.faults |= PINBALL_FAULT_BAD_FRAME;
      }
      log_frame(can_id, data, length, accepted);
      send_status();
      last_status_ms = now_ms;
    }

    if (pinball_watchdog_poll(&controller, now_ms)) {
      apply_outputs(0u);
      printf("WATCHDOG_RELEASE faults=0x%02x\n", controller.faults);
      send_status();
      last_status_ms = now_ms;
    }
    if ((uint32_t)(now_ms - last_status_ms) >= 100u) {
      if (xl2515_error_flags() != 0u) controller.faults |= PINBALL_FAULT_CAN;
      send_status();
      last_status_ms = now_ms;
    }
    if ((uint32_t)(now_ms - last_log_ms) >= 1000u) {
      printf("ALIVE state=%u rx=%u faults=0x%02x\n",
          controller.state, controller.rx_count, controller.faults);
      last_log_ms = now_ms;
    }
    tight_loop_contents();
  }
}
