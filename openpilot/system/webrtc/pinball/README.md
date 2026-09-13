# LAN pinball browser client

This dependency-free page receives the comma `road` and `wideRoad` WebRTC tracks and sends flipper state on the WebRTC data channel. It must be served from the same origin as `webrtcd`; the client posts its complete ICE-gathered offer to `/stream`.

## Control contract

Every input transition sends one JSON message immediately:

```json
{"type":"testJoystick","data":{"axes":[1,0],"buttons":[]}}
```

`axes[0]` is left and `axes[1]` is right. `1` means pressed and `0` means released. The page supports simultaneous presses, ignores keyboard repeat, and sends `[0,0]` when the data channel opens, the page loses focus, becomes hidden, unloads, or intentionally reconnects.

Controls are Left Arrow or `Z` for left, and Right Arrow or `/` for right. Pointer and touch controls are included.

## Serve and test

The embedding server should map this directory to `/` and keep the existing `POST /stream` handler on the same origin. A standalone local static server can render the UI but cannot negotiate unless it proxies `/stream`:

```sh
python3 -m http.server 8000 --directory system/webrtc/pinball
```

Run the deterministic protocol tests with:

```sh
node --test system/webrtc/pinball/protocol.test.js
```
