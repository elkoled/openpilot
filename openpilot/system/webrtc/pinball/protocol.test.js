"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const { createInputState, joystickMessage, sideForCode, streamRequest } = require("./protocol.js");

test("wire message has the exact testJoystick contract", () => {
  assert.deepEqual(JSON.parse(joystickMessage(true, false, true)), {
    type: "testJoystick", data: { axes: [1, 0, 1], buttons: [] },
  });
});

test("stream request matches the webrtcd StreamRequestBody contract", () => {
  assert.deepEqual(streamRequest("offer-sdp"), {
    sdp: "offer-sdp",
    cameras: ["road", "wideRoad"],
    enabled: true,
    bridge_services_in: ["testJoystick"],
    bridge_services_out: [],
  });
});

test("keyboard mappings reject unrelated keys", () => {
  assert.equal(sideForCode("ArrowLeft"), "left");
  assert.equal(sideForCode("KeyZ"), "left");
  assert.equal(sideForCode("ArrowRight"), "right");
  assert.equal(sideForCode("Slash"), "right");
  assert.equal(sideForCode("Space"), "start");
  assert.equal(sideForCode("Enter"), "start");
});

test("press/release is immediate, simultaneous, and deduplicated", () => {
  const sent = [];
  const state = createInputState((message) => sent.push(JSON.parse(message).data.axes));
  state.update("left", true);
  state.update("left", true);
  state.update("right", true);
  state.update("start", true);
  state.update("left", false);
  state.update("right", false);
  state.update("start", false);
  assert.deepEqual(sent, [[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]]);
});

test("forced disconnect release always emits neutral", () => {
  const sent = [];
  const state = createInputState((message) => sent.push(JSON.parse(message).data.axes));
  state.update("left", true);
  state.releaseAll(true);
  state.releaseAll(true);
  assert.deepEqual(sent, [[1, 0, 0], [0, 0, 0], [0, 0, 0]]);
  assert.deepEqual(state.snapshot(), { left: false, right: false, start: false });
});
