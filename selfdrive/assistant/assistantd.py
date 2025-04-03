#!/usr/bin/env python3
import os
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import threading
import requests

from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog
from openai import OpenAI
from ollama import chat
from pathlib import Path
from pydub import AudioSegment
import cereal.messaging as messaging

# ==== CONFIGURATION ====
LLM_HOST = "http://ollama.pixeldrift.win:11434"
TTS_HOST = "http://tts.pixeldrift.win"
FRAME_WIDTH = 1928
FRAMES_PER_SEC = 1      # Capture rate
BUFFER_SIZE = 1         # How many frames to collect before sending
REQUEST_TIMEOUT = 5     # Timeout for LLM requests in seconds
TTS_PLAYBACK_DELAY = 5  # Delay to complete playback of the TTS
USE_LOCAL_LLM = True
USE_LOCAL_TTS = True
LOCAL_LLM_MODEL = 'gemma3'

os.environ["OLLAMA_HOST"] = LLM_HOST

PROMPT_SYSTEM = (
    "Du bist ein echtzeit-Assistent, der Dashcam-Aufnahmen beobachtet und beschreibt, was visuell interessant oder relevant ist. "
    "Dein Ziel ist es, die Umgebung in einem klaren, gesprochenen Satz zu beschreiben. "
    "Konzentriere dich auf Dinge wie nahe Autos (Marke, Modell, Farbe, Kennzeichen), Fußgänger, Radfahrer, Tiere, Natur, Wetter und Verkehrsschilder. "
    "Versuche, sichtbare Schilder, Ortsschilder und Werbetafeln zu lesen und den tatsächlichen Text vorzulesen. "
    "Erwähne alles Ungewöhnliche, Überraschende oder Bemerkenswerte. "
    "Sprich natürlich, als würdest du dem Fahrer während der Fahrt erzählen, was passiert. "
    "Erwähne nicht, wenn es KEINE Fußgänger, Schilder oder Ähnliches gibt. "
    "Sag nicht die Dashcam, Dashcam, Dashcam-Aufnahme, Aufnahmen."
    "Sag nichts, dass hier eine beschreibung ist."
    "Ausschließlich die Beschreibung, keine Einleitung."
    "Sag dem Fahrer in einem einzigen Satz, was er tun und worauf er achten soll. "
)
# PROMPT_SYSTEM = (
#     "You are a real-time visual assistant that observes dashcam footage and describes what is visually interesting or relevant. "
#     "Your goal is to describe surroundings in one clear, spoken sentence. "
#     "Focus on things like nearby cars (make, model, color, license plate), pedestrians, cyclists, animals, nature, weather, and road signs. "
#     "Try to read and include the actual text on visible traffic signs, city limit signs, and billboards when possible. "
#     "Mention anything unusual, surprising, or worth noticing. "
#     "Speak naturally, as if you're narrating the drive to the person behind the wheel."
#     "Do not mention if there are NO pedestrians, signs or similar."
#     "Tell the driver what to do and what to look at in one single sentence."
#     "Do not say dashcam, dashcam footage, footage."
# )
# ========================


def get_vehicle_telemetry():
    """Get current vehicle telemetry data"""
    sm = messaging.SubMaster(['carState'])
    start = time.monotonic()
    while not sm.updated['carState']:
        sm.update(100)
        if time.monotonic() - start > 1.0:
            return "Vehicle telemetry not available yet. Use visual cues only."

    cs = sm['carState']
    speed_kph = round(cs.vEgoCluster * 3.6) if cs.vEgoCluster is not None else 0
    acceleration = round(cs.aEgo, 2) if cs.aEgo is not None else 0
    steering_angle = round(cs.steeringAngleDeg, 1)
    cruise_enabled = cs.cruiseState.enabled
    cruise_speed = round(cs.cruiseState.speed * 3.6) if cs.cruiseState.speed is not None else 0
    standstill = cs.standstill

    motion_state = "Das Fahrzeug steht." if standstill else f"Das Fahrzeug fährt mit {speed_kph} km/h."
    cruise_info = f"Tempomat aktiv bei {cruise_speed} km/h. " if cruise_enabled else ""

    return f"{motion_state}, Beschleunigung: {acceleration} m/s², Lenkwinkel: {steering_angle}°. {cruise_info}"
    # motion_state = "The vehicle is stationary." if standstill else f"The vehicle is moving at {speed_kph} km/h"
    # cruise_info = f"Cruise control active at {cruise_speed} km/h. " if cruise_enabled else ""

    # return f"{motion_state}, acceleration: {acceleration} m/s², steering angle: {steering_angle}°. {cruise_info}"

def build_prompt():
    """Build the user prompt with vehicle telemetry"""
    telemetry = get_vehicle_telemetry()
    return (
        f"Die folgenden {BUFFER_SIZE} Dashcam-Bilder wurden mit {FRAMES_PER_SEC} Bildern pro Sekunde aufgenommen "
        f"und zeigen die aktuelle Umgebung des Fahrzeugs. "
        "Beschreibe die visuell interessantesten und relevantesten Details in einem einzigen Satz den der Fahrer hören soll. "
        "Konzentriere dich auf nahe Fahrzeuge, Fußgänger, Radfahrer, Tiere, besondere Landschaften und insbesondere Texte auf Verkehrsschildern oder Werbetafeln. "
        f"{telemetry}"
    )
    # return (
    #     f"The following {BUFFER_SIZE} dashcam frames were captured at {FRAMES_PER_SEC} frames per second, "
    #     f"showing the vehicle's recent surroundings. "
    #     "Describe the most visually interesting and relevant details in a single spoken sentence. "
    #     "Focus on nearby cars, pedestrians, cyclists, animals, special scenery, and especially road signs or billboards. "
    #     f"{telemetry}"
    # )

def decode_nv12_to_jpeg(nv12_bytes, stride_y, width, height):
    """Convert NV12 format to JPEG"""
    try:
        y_size = stride_y * height
        y = np.frombuffer(nv12_bytes[:y_size], dtype=np.uint8).reshape((height, stride_y))[:, :width]

        uv_bytes = nv12_bytes[y_size:]
        uv_height = height // 2
        uv_stride = stride_y
        expected_uv_size = uv_height * uv_stride

        if len(uv_bytes) < expected_uv_size:
            cloudlog.error(f"[ASSISTANT] UV data too short: got {len(uv_bytes)}, expected {expected_uv_size}")
            return None

        uv_bytes = uv_bytes[:expected_uv_size]
        uv = np.frombuffer(uv_bytes, dtype=np.uint8).reshape((uv_height, uv_stride))
        u = uv[:, 0::2][:, :width // 2]
        v = uv[:, 1::2][:, :width // 2]

        u_up = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
        v_up = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

        min_h = min(y.shape[0], u_up.shape[0])
        crop_offset = 16  # crop top rows due to green lines
        y = y[crop_offset:min_h, :]
        u_up = u_up[crop_offset:min_h, :]
        v_up = v_up[crop_offset:min_h, :]

        # Convert to RGB
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
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        cloudlog.error(f"[ASSISTANT] decode_nv12_to_jpeg: {e}")
        return None

def llm_openai(images_b64, user_prompt):
    """Send images to OpenAI"""
    try:
        client = OpenAI(max_retries=0)
        full_prompt = f"{PROMPT_SYSTEM}\n\n{user_prompt}"

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

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": input_content
                }
            ],
            timeout=REQUEST_TIMEOUT
        )
        return response.output_text.strip()
    except Exception as e:
        cloudlog.error(f"[ASSISTANT] OpenAI request failed: {e}")
        return None

def llm_local(images_b64, user_prompt):
    try:
        image_bytes_list = [base64.b64decode(b64) for b64 in images_b64]

        response = chat(
            model=LOCAL_LLM_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': user_prompt,
                    'images': image_bytes_list,
                }
            ]
        )
        return response['message']['content'].strip()
    except Exception as e:
        cloudlog.error(f"[ASSISTANT] Local LLM (Ollama chat) failed: {e}")
        return None

def tts_openai(text):
    try:
        client = OpenAI()
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
            response_format="mp3"
        ) as response:
            mp3_bytes = response.read()
        save_audio(mp3_bytes, is_mp3=True)
        cloudlog.error(f"[ASSISTANT] OpenAI TTS ready: {text}")
    except Exception as e:
        cloudlog.error(f"[ASSISTANT] TTS failed: {e}")

def tts_local(text):
    try:
        response = requests.post(
            TTS_HOST,
            headers={"Content-Type": "text/plain"},
            data=text.encode("utf-8"),
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        if not response.content:
            cloudlog.error("[ASSISTANT] Local TTS: empty response.")
            return

        save_audio(response.content, is_mp3=False)
        cloudlog.error(f"[ASSISTANT] TTS done: {text}")
    except Exception as e:
        cloudlog.error(f"[ASSISTANT] Local TTS failed: {e}")


def save_audio(audio_bytes, is_mp3=True, output_path=Path("/tmp/play.wav")):
    try:
        audio = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3" if is_mp3 else "wav")
        audio = audio.set_frame_rate(48000).set_channels(1).set_sample_width(2)
        audio.export(output_path, format="wav")
        cloudlog.error("[ASSISTANT] Audio saved to WAV.")
    except Exception as e:
        cloudlog.error(f"[ASSISTANT] Audio conversion failed: {e}")

def main():
    """Main service loop"""
    config_realtime_process([0, 1, 2, 3], priority=5)
    vision_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)

    # Wait for camera connection
    while not vision_client.connect(False):
        time.sleep(0.1)

    cloudlog.error("[ASSISTANT] Connected to camera")

    frame_buffer = []
    last_result = ""
    last_capture_time = 0
    capture_interval = 1.0 / FRAMES_PER_SEC
    processing = False

    def process_frames(buffer_snapshot):
        nonlocal processing, last_result
        try:
            cloudlog.error(f"[ASSISTANT] Processing {len(buffer_snapshot)} frames")

            user_prompt = build_prompt()
            llm_func = llm_local if USE_LOCAL_LLM else llm_openai
            result = llm_func(buffer_snapshot, user_prompt)

            if not result or result == last_result:
                return  # skip empty or duplicate result
            cloudlog.error(f"[ASSISTANT] Result: {result}")
            tts_func = tts_local if USE_LOCAL_TTS else tts_openai
            tts_func(result)
            last_result = result
            # TODO: find other way
            time.sleep(TTS_PLAYBACK_DELAY)
        except Exception as e:
            cloudlog.error(f"[ASSISTANT] Processing error: {e}")
        finally:
            processing = False

    while True:
        try:
            buf = vision_client.recv()
            if buf is None:
                time.sleep(0.05)
                continue

            now = time.monotonic()
            if now - last_capture_time >= capture_interval:
                last_capture_time = now

                # Process frame
                buf_data = bytes(buf.data)
                encoded = decode_nv12_to_jpeg(buf_data, buf.stride, FRAME_WIDTH, buf.height)
                if not encoded:
                    continue

                # Manage buffer
                frame_buffer.append(encoded)
                if len(frame_buffer) > BUFFER_SIZE:
                    frame_buffer.pop(0)

            # Start processing if we have enough frames and not already processing
            if not processing and len(frame_buffer) == BUFFER_SIZE:
                processing = True
                buffer_snapshot = list(frame_buffer)
                threading.Thread(target=process_frames, args=(buffer_snapshot,)).start()

            time.sleep(0.01)
        except Exception as e:
            cloudlog.error(f"[ASSISTANT] Main loop error: {e}")
            time.sleep(1)  # Prevent tight error loops

if __name__ == "__main__":
    main()
