(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PinballProtocol = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const LEFT_KEYS = new Set(["ArrowLeft", "KeyZ"]);
  const RIGHT_KEYS = new Set(["ArrowRight", "Slash"]);

  function sideForCode(code) {
    if (LEFT_KEYS.has(code)) return "left";
    if (RIGHT_KEYS.has(code)) return "right";
    return null;
  }

  function joystickMessage(left, right) {
    return JSON.stringify({
      type: "testJoystick",
      data: { axes: [left ? 1 : 0, right ? 1 : 0], buttons: [] },
    });
  }

  function streamRequest(sdp) {
    return {
      sdp,
      cameras: ["road", "wideRoad"],
      enabled: true,
      bridge_services_in: ["testJoystick"],
      bridge_services_out: [],
    };
  }

  function createInputState(send) {
    let left = false;
    let right = false;

    function update(side, pressed) {
      const value = Boolean(pressed);
      if (side === "left") {
        if (left === value) return false;
        left = value;
      } else if (side === "right") {
        if (right === value) return false;
        right = value;
      } else {
        return false;
      }
      send(joystickMessage(left, right));
      return true;
    }

    function releaseAll(force) {
      if (!force && !left && !right) return false;
      left = false;
      right = false;
      send(joystickMessage(false, false));
      return true;
    }

    return {
      update,
      releaseAll,
      snapshot: () => ({ left, right }),
    };
  }

  return { sideForCode, joystickMessage, streamRequest, createInputState };
});
