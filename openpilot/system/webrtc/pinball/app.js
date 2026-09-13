(function () {
  "use strict";

  const status = document.querySelector("#status");
  const connectButton = document.querySelector("#connect");
  const videos = [document.querySelector("#road"), document.querySelector("#wideRoad")];
  const pads = [...document.querySelectorAll("[data-side]")];
  let peer = null;
  let channel = null;
  let reconnectTimer = null;
  let intentionalClose = false;

  function setStatus(kind, message) {
    status.dataset.kind = kind;
    status.textContent = message;
  }

  function send(payload) {
    if (channel && channel.readyState === "open") channel.send(payload);
  }

  const input = PinballProtocol.createInputState(send);

  function paintControls() {
    const state = input.snapshot();
    for (const pad of pads) pad.classList.toggle("pressed", state[pad.dataset.side]);
  }

  function setSide(side, pressed) {
    input.update(side, pressed);
    paintControls();
  }

  function releaseAll(force) {
    input.releaseAll(force);
    paintControls();
  }

  function waitForIceGathering(pc) {
    if (pc.iceGatheringState === "complete") return Promise.resolve();
    return new Promise((resolve) => {
      const changed = () => {
        if (pc.iceGatheringState === "complete") {
          pc.removeEventListener("icegatheringstatechange", changed);
          resolve();
        }
      };
      pc.addEventListener("icegatheringstatechange", changed);
    });
  }

  function scheduleReconnect(message) {
    if (intentionalClose || reconnectTimer) return;
    setStatus("error", `${message} Reconnecting…`);
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, 2000);
  }

  function closePeer(intentional) {
    intentionalClose = intentional;
    releaseAll(true);
    const oldChannel = channel;
    const oldPeer = peer;
    channel = null;
    peer = null;
    if (oldChannel) oldChannel.close();
    if (oldPeer) oldPeer.close();
    for (const video of videos) video.srcObject = null;
  }

  async function connect() {
    if (peer) closePeer(true);
    intentionalClose = false;
    connectButton.disabled = true;
    setStatus("connecting", "Connecting to comma…");

    const pc = new RTCPeerConnection({ iceServers: [] });
    peer = pc;
    let trackIndex = 0;

    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("video", { direction: "recvonly" });
    pc.ontrack = (event) => {
      const target = videos[Math.min(trackIndex++, videos.length - 1)];
      target.srcObject = event.streams[0] || new MediaStream([event.track]);
      target.play().catch(() => {});
    };

    const dc = pc.createDataChannel("data");
    channel = dc;
    dc.onopen = () => {
      if (channel !== dc) return;
      releaseAll(true);
      setStatus("connected", "LIVE — controls armed");
      connectButton.disabled = false;
      connectButton.textContent = "Reconnect";
    };
    dc.onclose = () => {
      if (channel === dc) scheduleReconnect("Control channel closed.");
    };
    dc.onerror = () => {
      if (channel === dc) scheduleReconnect("Control channel failed.");
    };
    dc.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === "disconnect") scheduleReconnect(String(message.data || "Comma disconnected."));
      } catch (_) { /* Ignore messages unrelated to this demo. */ }
    };
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === "failed" || pc.connectionState === "disconnected") {
        scheduleReconnect(`WebRTC ${pc.connectionState}.`);
      }
    };

    try {
      await pc.setLocalDescription(await pc.createOffer());
      await waitForIceGathering(pc);
      const response = await fetch("/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(PinballProtocol.streamRequest(pc.localDescription.sdp)),
      });
      const answer = await response.json();
      if (!response.ok || !answer.sdp) throw new Error(answer.message || answer.error || `HTTP ${response.status}`);
      await pc.setRemoteDescription(answer);
      setStatus("connecting", "WebRTC negotiated; waiting for video and controls…");
    } catch (error) {
      if (peer === pc) {
        closePeer(false);
        connectButton.disabled = false;
        scheduleReconnect(error.message || "Connection failed.");
      }
    }
  }

  document.addEventListener("keydown", (event) => {
    const side = PinballProtocol.sideForCode(event.code);
    if (!side) return;
    event.preventDefault();
    if (!event.repeat) setSide(side, true);
  });
  document.addEventListener("keyup", (event) => {
    const side = PinballProtocol.sideForCode(event.code);
    if (!side) return;
    event.preventDefault();
    setSide(side, false);
  });
  for (const pad of pads) {
    const side = pad.dataset.side;
    pad.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      pad.setPointerCapture(event.pointerId);
      setSide(side, true);
    });
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
      pad.addEventListener(name, () => setSide(side, false));
    }
  }
  window.addEventListener("blur", () => releaseAll(false));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) releaseAll(false);
  });
  window.addEventListener("pagehide", () => {
    intentionalClose = true;
    releaseAll(true);
  });
  connectButton.addEventListener("click", connect);
  connect();
})();
