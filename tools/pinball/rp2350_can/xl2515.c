#include "xl2515.h"

#include "hardware/spi.h"
#include "pico/stdlib.h"

#define CAN_SPI spi1
#define CAN_SCK 10u
#define CAN_MOSI 11u
#define CAN_MISO 12u
#define CAN_CS 9u
#define CAN_INT 8u

#define CMD_RESET 0xc0u
#define CMD_READ 0x03u
#define CMD_WRITE 0x02u
#define CMD_BIT_MODIFY 0x05u

#define REG_CANSTAT 0x0eu
#define REG_CANCTRL 0x0fu
#define REG_CNF3 0x28u
#define REG_CNF2 0x29u
#define REG_CNF1 0x2au
#define REG_CANINTE 0x2bu
#define REG_CANINTF 0x2cu
#define REG_EFLG 0x2du
#define REG_TXB0CTRL 0x30u
#define REG_TXB0SIDH 0x31u
#define REG_TXB0SIDL 0x32u
#define REG_TXB0DLC 0x35u
#define REG_TXB0D0 0x36u
#define REG_RXB0CTRL 0x60u
#define REG_RXB0SIDH 0x61u
#define REG_RXB0SIDL 0x62u
#define REG_RXB0DLC 0x65u
#define REG_RXB0D0 0x66u
#define REG_RXF0SIDH 0x00u
#define REG_RXF0SIDL 0x01u
#define REG_RXM0SIDH 0x20u
#define REG_RXM0SIDL 0x21u

static inline void select_chip(void) { gpio_put(CAN_CS, 0); }
static inline void deselect_chip(void) { gpio_put(CAN_CS, 1); }

static void write_register(uint8_t reg, uint8_t value) {
  const uint8_t bytes[3] = {CMD_WRITE, reg, value};
  select_chip();
  spi_write_blocking(CAN_SPI, bytes, 3);
  deselect_chip();
}

static uint8_t read_register(uint8_t reg) {
  const uint8_t command[2] = {CMD_READ, reg};
  uint8_t value;
  select_chip();
  spi_write_blocking(CAN_SPI, command, 2);
  spi_read_blocking(CAN_SPI, 0, &value, 1);
  deselect_chip();
  return value;
}

static void write_registers(uint8_t reg, const uint8_t *values, uint8_t count) {
  const uint8_t command[2] = {CMD_WRITE, reg};
  select_chip();
  spi_write_blocking(CAN_SPI, command, 2);
  spi_write_blocking(CAN_SPI, values, count);
  deselect_chip();
}

static void read_registers(uint8_t reg, uint8_t *values, uint8_t count) {
  const uint8_t command[2] = {CMD_READ, reg};
  select_chip();
  spi_write_blocking(CAN_SPI, command, 2);
  spi_read_blocking(CAN_SPI, 0, values, count);
  deselect_chip();
}

static void bit_modify(uint8_t reg, uint8_t mask, uint8_t value) {
  const uint8_t command[4] = {CMD_BIT_MODIFY, reg, mask, value};
  select_chip();
  spi_write_blocking(CAN_SPI, command, 4);
  deselect_chip();
}

bool xl2515_init_500k(void) {
  spi_init(CAN_SPI, 10u * 1000u * 1000u);
  gpio_set_function(CAN_SCK, GPIO_FUNC_SPI);
  gpio_set_function(CAN_MOSI, GPIO_FUNC_SPI);
  gpio_set_function(CAN_MISO, GPIO_FUNC_SPI);
  gpio_init(CAN_CS);
  gpio_set_dir(CAN_CS, GPIO_OUT);
  deselect_chip();
  gpio_init(CAN_INT);
  gpio_set_dir(CAN_INT, GPIO_IN);
  gpio_pull_up(CAN_INT);

  select_chip();
  const uint8_t reset = CMD_RESET;
  spi_write_blocking(CAN_SPI, &reset, 1);
  deselect_chip();
  sleep_ms(10);

  /* Waveshare's official 16 MHz oscillator table: 500 kbit/s. */
  write_register(REG_CNF1, 0x00u);
  write_register(REG_CNF2, 0x92u);
  write_register(REG_CNF3, 0x02u);

  /* Exact standard-ID filter for 0x200 in RX buffer 0. */
  write_register(REG_RXF0SIDH, (uint8_t)(0x200u >> 3));
  write_register(REG_RXF0SIDL, (uint8_t)((0x200u & 7u) << 5));
  write_register(REG_RXM0SIDH, 0xffu);
  write_register(REG_RXM0SIDL, 0xe0u);
  write_register(REG_RXB0CTRL, 0x00u); /* filters enabled; no unread RXB1 rollover */
  write_register(REG_CANINTF, 0u);
  write_register(REG_CANINTE, 0x01u);
  write_register(REG_CANCTRL, 0u); /* normal mode */

  const absolute_time_t deadline = make_timeout_time_ms(20);
  while ((read_register(REG_CANSTAT) & 0xe0u) != 0u) {
    if (absolute_time_diff_us(get_absolute_time(), deadline) <= 0) return false;
  }
  return true;
}

bool xl2515_receive(uint16_t *standard_id, uint8_t data[8], uint8_t *length) {
  if (gpio_get(CAN_INT) != 0 && (read_register(REG_CANINTF) & 0x01u) == 0u) return false;
  if ((read_register(REG_CANINTF) & 0x01u) == 0u) return false;
  const uint8_t sidh = read_register(REG_RXB0SIDH);
  const uint8_t sidl = read_register(REG_RXB0SIDL);
  *standard_id = (uint16_t)(((uint16_t)sidh << 3) | (sidl >> 5));
  *length = (uint8_t)(read_register(REG_RXB0DLC) & 0x0fu);
  if (*length > 8u) *length = 8u;
  read_registers(REG_RXB0D0, data, *length);
  bit_modify(REG_CANINTF, 0x01u, 0u);
  return true;
}

bool xl2515_send(uint16_t standard_id, const uint8_t *data, uint8_t length) {
  if (standard_id > 0x7ffu || length > 8u) return false;
  const absolute_time_t deadline = make_timeout_time_ms(3);
  while ((read_register(REG_TXB0CTRL) & 0x08u) != 0u) {
    if (absolute_time_diff_us(get_absolute_time(), deadline) <= 0) return false;
  }
  write_register(REG_TXB0SIDH, (uint8_t)(standard_id >> 3));
  write_register(REG_TXB0SIDL, (uint8_t)((standard_id & 7u) << 5));
  write_register(REG_TXB0DLC, length);
  write_registers(REG_TXB0D0, data, length);
  bit_modify(REG_TXB0CTRL, 0x08u, 0x08u);
  return true;
}

uint8_t xl2515_error_flags(void) { return read_register(REG_EFLG); }
