#!/usr/bin/env python3
import os
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import threading

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
#FRAME_HEIGHT = 1662 # SIM resolution: 1208, Comma3x resolution: 1662
FRAMES_PER_SEC = 0.5        # Capture rate
BUFFER_SIZE = 5           # How many frames to collect before sending
PROMPT_SYSTEM = (
    "You are a real-time visual assistant that observes dashcam footage and describes what is visually interesting or relevant. "
    "Your goal is to describe surroundings in one clear, spoken sentence. "
    "Focus on things like nearby cars (make, model, color), pedestrians, cyclists, animals, nature, weather, and road signs. "
    "Try to read and include the actual text on visible traffic signs, city limit signs, and billboards when possible. "
    "Mention anything unusual, surprising, or worth noticing. "
    "Speak naturally, as if you're narrating the drive to the person behind the wheel."
    "Do not mention if there are NO pedestrians, signs or similar."
    "Tell the driver what to do and what to look at in one single sentence."
)
# ========================

os.environ["OLLAMA_HOST"] = OLLAMA_HOST
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

openai_client = OpenAI()

def build_prompt_user(buffer_size, fps):
  sm = messaging.SubMaster(['carState'])
  start = time.monotonic()
  while not sm.updated['carState']:
    sm.update(100)
    if time.monotonic() - start > 1.0:
        return "Vehicle telemetry not available yet. Use visual cues only."

  cs = sm['carState']

  # Fallbacks
  speed_kph = round(cs.vEgoCluster * 3.6) if cs.vEgoCluster is not None else 0
  acceleration = round(cs.aEgo, 2) if cs.aEgo is not None else 0
  steering_angle = round(cs.steeringAngleDeg, 1)
  steering_torque = round(cs.steeringTorque, 2)
  steering_rate = round(cs.steeringRateDeg, 2)
  yaw_rate = round(cs.yawRate, 2)
  cruise_enabled = cs.cruiseState.enabled
  cruise_speed = round(cs.cruiseState.speed * 3.6) if cs.cruiseState.speed is not None else 0
  cruise_standstill = cs.cruiseState.standstill
  gas_pct = round(cs.gas * 100) if cs.gas is not None else 0
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
    f"The following {buffer_size} dashcam frames were captured at {fps} frames per second, "
    f"showing the vehicle's recent surroundings. "
    "Describe the most visually interesting and relevant details in a single spoken sentence. "
    "Focus on nearby cars (make, model, color), pedestrians, cyclists, animals, special scenery, and especially road signs or billboards — read the actual text if visible. "
    "Imagine you are a real-time co-driver, pointing out what stands out during the drive. "
    + additional_info
  )


def decode_nv12_to_jpeg(nv12_bytes, stride_y, width, height):
    try:
        cloudlog.error(f"[ASSISTANT] decode_nv12_to_jpeg: stride={stride_y}, width={width}, height={height}, buffer={len(nv12_bytes)}")

        y_size = stride_y * height
        y = np.frombuffer(nv12_bytes[:y_size], dtype=np.uint8).reshape((height, stride_y))[:, :width]

        uv_bytes = nv12_bytes[y_size:]
        uv_height = height // 2
        uv_stride = stride_y  # NV12 UV stride == Y stride
        expected_uv_size = uv_height * uv_stride

        if len(uv_bytes) < expected_uv_size:
            cloudlog.error(f"[ASSISTANT] UV data too short: got {len(uv_bytes)}, expected {expected_uv_size}")
            return None
        if len(uv_bytes) > expected_uv_size:
            cloudlog.error(f"[ASSISTANT] Trimming padded UV: got {len(uv_bytes)}, expected {expected_uv_size}")
            uv_bytes = uv_bytes[:expected_uv_size]

        uv = np.frombuffer(uv_bytes, dtype=np.uint8).reshape((uv_height, uv_stride))
        u = uv[:, 0::2][:, :width // 2]
        v = uv[:, 1::2][:, :width // 2]

        u_up = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
        v_up = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

        min_h = min(y.shape[0], u_up.shape[0])
        crop_offset = 16  # number of top rows to crop due to green lines
        y = y[crop_offset:min_h, :]
        u_up = u_up[crop_offset:min_h, :]
        v_up = v_up[crop_offset:min_h, :]

        y_f = y.astype(np.float32)
        u_f = u_up.astype(np.float32) - 128
        v_f = v_up.astype(np.float32) - 128

        r = y_f + 1.402 * v_f
        g = y_f - 0.344136 * u_f - 0.714136 * v_f
        b = y_f + 1.772 * u_f

        rgb = np.stack([
            np.clip(r, 0, 255).astype(np.uint8),
            np.clip(g, 0, 255).astype(np.uint8),
            np.clip(b, 0, 255).astype(np.uint8),
        ], axis=2)

        img = Image.fromarray(rgb)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=50)
        cloudlog.error(f"[ASSISTANT] decode_nv12_to_jpeg: successfully encoded JPEG ({len(buf.getvalue())} bytes)")
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        cloudlog.exception(f"[ASSISTANT] decode_nv12_to_jpeg: {e}")
        return None

def send_to_ollama(images_b64, user_prompt):
    try:
        response = chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": user_prompt, "images": images_b64}
            ]
        )
        return response.message.content.strip()
    except Exception as e:
        cloudlog.exception(f"[ASSISTANT] send_to_model: {e}")
        return ""

def send_to_openai(images_b64, user_prompt):
    try:
        # Combine system prompt with user prompt in the first message
        full_prompt = f"{PROMPT_SYSTEM}\n\n{user_prompt}"

        # Save images to debug folder
        debug_folder = Path("/tmp/assistant_images")
        debug_folder.mkdir(parents=True, exist_ok=True)
        for idx, img_b64 in enumerate(images_b64):
            img_data = base64.b64decode(img_b64)
            img_path = debug_folder / f"frame_{idx:02}.jpg"
            with open(img_path, "wb") as f:
                f.write(img_data)
            cloudlog.error(f"[ASSISTANT] Saved debug image to {img_path}")

        input_content = [
            {"type": "input_text", "text": full_prompt}
        ] + [
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{img}",
                "detail": "low"
            }
            for img in images_b64
        ]

        cloudlog.error(f"[ASSISTANT] Sending {len(images_b64)} images using OpenAI responses API")

        response = openai_client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": input_content
                }
            ]
        )

        return response.output_text.strip()
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
    vision_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
    while not vision_client.connect(False):
        time.sleep(0.1)

    last_result = ""
    frame_buffer = []
    last_capture_time = 0
    capture_interval = 1.0 / FRAMES_PER_SEC
    processing_thread = None

    def process_frames(buffer_snapshot):
        nonlocal last_result
        try:
            user_prompt = build_prompt_user(BUFFER_SIZE, FRAMES_PER_SEC)
            cloudlog.info(f"[ASSISTANT] Processing {len(buffer_snapshot)} frames")
            cloudlog.error(user_prompt)
            if LLM_BACKEND == "ollama":
                result = send_to_ollama(buffer_snapshot, user_prompt)
            elif LLM_BACKEND == "openai":
                result = send_to_openai(buffer_snapshot, user_prompt)
            else:
                raise ValueError(f"[ASSISTANT] Unknown LLM_BACKEND: {LLM_BACKEND}")

            if result and result != last_result:
                cloudlog.error(f"[ASSISTANT] {result}")
                play_tts_audio(result)
                last_result = result
        except Exception as e:
            cloudlog.exception("[ASSISTANT] process_frames failed")

    while True:
        buf = vision_client.recv()
        if buf is None:
            time.sleep(0.05)
            continue

        now = time.monotonic()
        if now - last_capture_time >= capture_interval:
            last_capture_time = now
            buf_data = bytes(buf.data)
            stride = buf.stride
            encoded = decode_nv12_to_jpeg(buf_data, stride, FRAME_WIDTH, buf.height)
            if not encoded:
                cloudlog.error("[ASSISTANT] Frame encoding failed — skipping")
                continue

            frame_buffer.append(encoded)
            if len(frame_buffer) > BUFFER_SIZE:
                frame_buffer.pop(0)

        if processing_thread is None or not processing_thread.is_alive():
            if len(frame_buffer) == BUFFER_SIZE:
                buffer_snapshot = list(frame_buffer)
                processing_thread = threading.Thread(target=process_frames, args=(buffer_snapshot,))
                processing_thread.start()

        time.sleep(0.01)

def main():
    assistantd()

if __name__ == "__main__":
    main()
