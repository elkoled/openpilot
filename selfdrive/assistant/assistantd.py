#!/usr/bin/env python3
import os
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog
from ollama import chat

# ==== CONFIGURATION ====
OLLAMA_HOST = "http://ollama.pixeldrift.win:11434"
MODEL_NAME = "gemma3"
FRAME_WIDTH = 1928
FRAME_HEIGHT = 1208
FRAMES_PER_SEC = 1        # Capture rate
BUFFER_SIZE = 5           # How many frames to collect before sending
PROMPT_SYSTEM = (
    "You are a real-time driving assistant. Given a sequence of dashcam images taken over several seconds, "
    "generate exactly one short spoken sentence for the driver. "
    "Focus on what is visible: road signs, speed limits, pedestrians, vehicles, curves, traffic lights, weather, or hazards. "
    "Do not include explanations or summaries. Only output the sentence the driver should hear — concise and relevant."
)
PROMPT_USER = (
    f"The following {BUFFER_SIZE} dashcam frames were captured at a rate of {FRAMES_PER_SEC} frames per second, "
    f"representing the last {int(BUFFER_SIZE / FRAMES_PER_SEC)} seconds of driving. "
    "Generate one clear and useful sentence the driver should hear."
)
# ========================

os.environ["OLLAMA_HOST"] = OLLAMA_HOST

def decode_nv12_to_jpeg(nv12_bytes):
    try:
        y_size = FRAME_WIDTH * FRAME_HEIGHT
        uv_size = y_size // 2
        if len(nv12_bytes) != y_size + uv_size:
            return None

        y = np.frombuffer(nv12_bytes[:y_size], dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH))
        uv = np.frombuffer(nv12_bytes[y_size:], dtype=np.uint8).reshape((FRAME_HEIGHT // 2, FRAME_WIDTH))
        nv12 = np.vstack((y, uv))

        # Convert NV12 to RGB using PIL
        img = Image.fromarray(nv12, mode="L").convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        cloudlog.exception(f"decode_nv12_to_jpeg: {e}")
        return None

def send_to_model(images_b64):
    try:
        response = chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": PROMPT_USER, "images": images_b64}
            ]
        )
        return response.message.content.strip()
    except Exception as e:
        cloudlog.exception(f"send_to_model: {e}")
        return ""

def assistantd():
    config_realtime_process([0, 1, 2, 3], priority=5)
    client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
    while not client.connect(False):
        time.sleep(0.1)

    last_result = ""
    frame_buffer = []
    last_capture_time = 0
    capture_interval = FRAMES_PER_SEC

    while True:
        buf = client.recv()
        if buf is None:
            continue

        now = time.monotonic()
        if now - last_capture_time >= capture_interval:
            last_capture_time = now
            encoded = decode_nv12_to_jpeg(bytes(buf.data))
            if encoded:
                frame_buffer.append(encoded)

        if len(frame_buffer) >= BUFFER_SIZE:
            result = send_to_model(frame_buffer)
            if result and result != last_result:
                cloudlog.warning(f"[DASHCAM-ASSIST] {result}")
                last_result = result
            frame_buffer.clear()

        time.sleep(0.05)

def main():
    assistantd()

if __name__ == "__main__":
    main()
