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
from openai import OpenAI
from pathlib import Path

# ==== CONFIGURATION ====
LLM_BACKEND = "openai"  # Options: "ollama", "openai"
# OPENAI_API_KEY = ""
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
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

openai_client = OpenAI()

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
        cloudlog.exception(f"[ASSISTANT] decode_nv12_to_jpeg: {e}")
        return None

def send_to_ollama(images_b64):
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
        cloudlog.exception(f"[ASSISTANT] send_to_model: {e}")
        return ""

def send_to_openai(images_b64):
    try:
        content = [{"type": "text", "text": PROMPT_USER}]
        for image_b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "auto"
                }
            })

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": content}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        cloudlog.exception(f"[ASSISTANT] send_to_openai: {e}")
        return ""

def play_tts_audio(text, file_path="/data/openpilot/selfdrive/assets/sounds/tts.wav"):
    try:
        with OpenAI().audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
            response_format="wav",
        ) as response:
            response.stream_to_file(Path(file_path))

        # Create a flag file to notify soundd
        Path("/tmp/play_tts.flag").touch()

        print(f"[ASSISTANT] TTS saved: {text}")
    except Exception as e:
        print(f"[ASSISTANT] play_tts_audio failed: {e}")

def assistantd():
    config_realtime_process([0, 1, 2, 3], priority=5)
    vision_client  = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
    while not vision_client .connect(False):
        time.sleep(0.1)

    last_result = ""
    frame_buffer = []
    last_capture_time = 0
    capture_interval = 1.0 / FRAMES_PER_SEC
    waiting_for_response = False
    response_start_time = None
    response_timeout_sec = 20  # maximum time to wait for model

    while True:
        buf = vision_client .recv()
        if buf is None:
            time.sleep(0.05)
            continue

        now = time.monotonic()

        # If we're waiting too long for a response, reset
        if waiting_for_response and now - response_start_time > response_timeout_sec:
            cloudlog.warning("[ASSISTANT] Model response timeout — skipping")
            waiting_for_response = False
            frame_buffer.clear()

        # Only collect frames if not waiting
        if not waiting_for_response and now - last_capture_time >= capture_interval:
            last_capture_time = now
            encoded = decode_nv12_to_jpeg(bytes(buf.data))
            if encoded:
                frame_buffer.append(encoded)

        if not waiting_for_response and len(frame_buffer) >= BUFFER_SIZE:
            response_start_time = now
            waiting_for_response = True
            try:
                cloudlog.info(f"[ASSISTANT] Captured {len(frame_buffer)} frames, starting model inference")
                if LLM_BACKEND == "ollama":
                    result = send_to_ollama(frame_buffer)
                elif LLM_BACKEND == "openai":
                    result = send_to_openai(frame_buffer)
                else:
                    raise ValueError(f"[ASSISTANT] Unknown LLM_BACKEND: {LLM_BACKEND}")

                if result and result != last_result:
                    cloudlog.warning(f"[ASSISTANT] {result}")
                    play_tts_audio(result)
                    last_result = result
            except Exception as e:
                cloudlog.exception("[ASSISTANT] send_to_model failed")
            finally:
                waiting_for_response = False
                frame_buffer.clear()

        time.sleep(0.05)

def main():
    assistantd()

if __name__ == "__main__":
    main()
