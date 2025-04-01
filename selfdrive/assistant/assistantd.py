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
from pydub import AudioSegment
import cereal.messaging as messaging

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
    "You are a real-time driving assistant. Your goal is to help the driver stay safe, alert, and informed "
    "while driving. You receive a short sequence of dashcam images and live vehicle telemetry. "
    "Generate exactly one spoken sentence that is clear, useful, and timely. "
    "Prioritize important road information such as: speed limit signs, traffic signs, pedestrians, road curvature, "
    "traffic lights, vehicle distance, obstacles, and hazards. "
    "Avoid summarizing or describing — speak directly as if you're giving driving instructions. "
    "The sentence should be what the driver hears in the moment, not a report or explanation."
)
PROMPT_USER = ""
# PROMPT_USER = (
#     f"The following {BUFFER_SIZE} dashcam frames were captured at a rate of {FRAMES_PER_SEC} frames per second, "
#     f"representing the last {int(BUFFER_SIZE / FRAMES_PER_SEC)} seconds of driving. "
#     "Generate one clear and useful sentence the driver should hear."
# )
# ========================

os.environ["OLLAMA_HOST"] = OLLAMA_HOST
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

openai_client = OpenAI()

def build_prompt_user(buffer_size, fps):
  sm = messaging.SubMaster(['carState'])
  sm.update(100)  # non-blocking update

  cs = sm['carState']

  # Fallbacks
  speed_kph = round(cs.vEgoCluster * 3.6) if cs.vEgoCluster else 0
  acceleration = round(cs.aEgo, 2) if cs.aEgo else 0
  steering_angle = round(cs.steeringAngleDeg, 1)
  steering_torque = round(cs.steeringTorque, 2)
  steering_rate = round(cs.steeringRateDeg, 2)
  yaw_rate = round(cs.yawRate, 2)
  cruise_enabled = cs.cruiseState.enabled
  cruise_speed = round(cs.cruiseState.speed * 3.6) if cs.cruiseState.speed else 0
  cruise_standstill = cs.cruiseState.standstill
  gas_pct = round(cs.gas * 100) if cs.gas else 0
  brake_hold = cs.brakeHoldActive
  esp_disabled = cs.espDisabled
  steering_override = cs.steeringPressed
  standstill = cs.standstill

  wheel_speeds = cs.wheelSpeeds
  avg_wheel_speed = round((wheel_speeds.fl + wheel_speeds.fr + wheel_speeds.rl + wheel_speeds.rr) / 4 * 3.6, 1)

  motion_state = "The vehicle is stationary." if standstill else f"The vehicle is moving at {speed_kph} km/h"

  additional_info = (
    f"{motion_state} Avg wheel speed: {avg_wheel_speed} km/h, acceleration: {acceleration} m/s², "
    f"yaw rate: {yaw_rate}°/s, steering angle: {steering_angle}°, rate: {steering_rate}°/s, torque: {steering_torque}, "
    f"gas: {gas_pct}%. "
    f"{'Cruise control active at ' + str(cruise_speed) + ' km/h. ' if cruise_enabled else ''}"
    f"{'Cruise is waiting to resume from standstill. ' if cruise_standstill else ''}"
    f"{'Brake hold is active. ' if brake_hold else ''}"
    f"{'Driver is overriding steering. ' if steering_override else ''}"
    f"{'Warning: Stability control (ESP) is disabled. ' if esp_disabled else ''}"
  )

  return (
    f"The following {buffer_size} dashcam frames were captured at a rate of {fps} frames per second, "
    f"representing the last {int(buffer_size / fps)} seconds of driving. "
    "You are a real-time co-pilot and assistant, helping the driver with concise spoken messages. "
    "Focus **especially** on visible speed limit signs, street signs, curves, pedestrians, traffic lights, and any potential hazards. "
    "Give safety-relevant and regulation-relevant messages first. Avoid repetition. "
    + additional_info +
    "Generate exactly one spoken sentence based only on the telemetry and what you see. Do not add explanations."
  )



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

def play_tts_audio(text):
    try:
        temp_mp3 = Path("/tmp/tts.mp3")
        final_wav = Path("/tmp/play.wav")

        with OpenAI().audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
            response_format="mp3"
        ) as response:
            response.stream_to_file(temp_mp3)

        # Resample from 24kHz to 48kHz mono WAV using pydub
        audio = AudioSegment.from_file(temp_mp3)
        audio = audio.set_frame_rate(48000).set_channels(1).set_sample_width(2)
        audio.export(final_wav, format="wav")

        print(f"[ASSISTANT] TTS ready at {final_wav}: {text}")
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
                PROMPT_USER = build_prompt_user(BUFFER_SIZE, FRAMES_PER_SEC)
                cloudlog.warning(PROMPT_USER)
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
