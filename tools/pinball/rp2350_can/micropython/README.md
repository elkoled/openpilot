# MicroPython variant (not deployed)

Copy `main.py`, `protocol.py`, and `xl2515.py` to the root of a MicroPython
filesystem on the Waveshare RP2350-CAN. `main.py` then starts automatically.

This is a source-equivalent rewrite of the working C firmware. It uses the same
XL2515 SPI1 pins and 500-kbit/s timing, CAN IDs, servo pins/endpoints, direct
button-state semantics, status frames, 200 ms loss-of-communications release,
and USB serial diagnostics. It adds no press pulse or hold delay.

It has deliberately not been copied to or flashed onto the connected board.

Run the host-side protocol tests with:

```sh
python3 -m unittest discover tools/pinball/rp2350_can/micropython/tests
```
