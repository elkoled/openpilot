# RP2350-CAN pinball servo firmware

Minimal firmware for the Waveshare RP2350-CAN board. It receives button state
from the comma/Panda at 500 kbit/s and drives both flipper servos immediately.
There is no artificial hold or bump delay.

## Wiring and outputs

The onboard XL2515 uses SPI1: INT GP8, CS GP9, SCK GP10, MOSI GP11, MISO GP12.
The firmware uses GP14 for the left servo and GP15 for the right servo. Both run
at 333 Hz (3003 us period). GP4 drives the START solenoid directly: HIGH only
while START is held, LOW on release.

| Output | Released | Pressed |
| --- | ---: | ---: |
| GP14 left | 1570 us | 1770 us |
| GP15 right | 1420 us | 1220 us |

The servo supply must be external and appropriately sized. Join grounds. Connect
CAN-H to CAN-H and CAN-L to CAN-L. A two-node bus needs 120 ohm termination at
each physical end; the board has a termination selector.

## CAN contract

All frames are standard 11-bit CAN with exactly 8 data bytes.

Command `0x200`:

| Byte | Meaning |
| ---: | --- |
| 0..1 | magic `50 42` |
| 2 | protocol version `01` |
| 3 | rolling sequence, increment for every command |
| 4 | state: bit 0 left, bit 1 right, bit 2 START; other bits must be zero |
| 5 | bitwise inverse of byte 4 |
| 6 | reserved; must be zero |
| 7 | CRC-8/SAE-J1850 over bytes 0..6 (poly 1D, init/xorout FF) |

Status/ack `0x201`:

| Byte | Meaning |
| ---: | --- |
| 0..1 | magic `50 42` |
| 2 | protocol version `01` |
| 3 | last accepted sequence |
| 4 | applied state in bits 0..1; watchdog-active flag in bit 7 |
| 5..6 | accepted-command count, unsigned little-endian 16-bit |
| 7 | CRC-8/SAE-J1850 over bytes 0..6 |

Commands must advance modulo 256 by 1..127. A duplicate, rollback, bad magic,
reserved state bit/byte, inverse mismatch, or CRC mismatch is rejected and does not
refresh the watchdog. After 200 ms without a newly accepted command, both outputs
return to release and sequence history is cleared. A subsequent valid sequence of
any value may re-arm control, allowing a restarted sender to resynchronize.

Golden command for sequence 42, left pressed: `50 42 01 2A 01 FE 00 A8`.

## Test and build

Run the host gate and deterministic eval:

```sh
tools/pinball/rp2350_can/tests/run.sh
python3 tools/pinball/rp2350_can/evals/protocol_eval.py
```

Build with Raspberry Pi Pico SDK 2.x:

```sh
cd tools/pinball/rp2350_can
PICO_SDK_PATH=/path/to/pico-sdk cmake -S . -B build
cmake --build build -j
```

Flash `build/pinball_rp2350_can.uf2` while holding BOOTSEL. Outputs start in
release position before CAN initialization. The XL2515 driver is readable C
source derived from Waveshare's official RP2350-CAN demo; no vendor binary is
used. The upstream demo used a polling loop with a potentially unbounded receive
wait; this version uses bounded reads and hardware filtering for command ID 0x200.
