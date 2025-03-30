#!/usr/bin/env python3
import os
os.environ["OLLAMA_HOST"] = "http://ollama.pixeldrift.win:11434"

import time
import base64
from ollama import chat
import cereal.messaging as messaging
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog

def encode_image_bytes(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode()

def process_frames(image_bytes_list):
    cloudlog.warning(f"PROCESSING IMAGES 1")
    if len(image_bytes_list) < 5:
        return ""

    encoded_images = [encode_image_bytes(img) for img in image_bytes_list[:5]]
    cloudlog.warning(f"PROCESSING IMAGES 2")

    response = chat(
        model='gemma3',
        messages=[
            {
                'role': 'system',
                'content': (
                    "You are a driving assistant that provides one short spoken sentence to the driver. "
                    "Focus on visible hazards, road signs, lights, pedestrians, and weather. No explanations."
                )
            },
            {
                'role': 'user',
                'content': (
                    "Give only one short sentence that should be played to the driver, based on these dashcam frames."
                ),
                'images': encoded_images,
            }
        ]
    )
    cloudlog.warning(f"PROCESSING IMAGES 3")
    return response.message.content.strip()

def assistd_thread():
    config_realtime_process([0, 1, 2, 3], priority=5)

    sm = messaging.SubMaster(['roadCameraState'])
    cloudlog.warning(f"ASSISTANTD: {sm.data}")

    frame_buffer = []
    last_result = ""

    while True:
        sm.update()
        if sm.updated['roadCameraState']:
            img_data = sm['roadCameraState'].image
            frame_buffer.append(img_data)
            cloudlog.warning(f"ASSISTANTD image: {img_data}, Framebuffer: {len(frame_buffer)}")

            if len(frame_buffer) >= 5:

                try:
                    result = process_frames(frame_buffer)
                    if result and result != last_result:
                        print(f"[DASHCAM-ASSIST] {result}")
                        cloudlog.warning(f"[DASHCAM-ASSIST] {result}")
                        last_result = result
                    frame_buffer.clear()
                except Exception as e:
                    print(f"[DASHCAM-ASSIST] Error: {e}")
                    cloudlog.warning(f"[DASHCAM-ASSIST] Error: {e}")
                    frame_buffer.clear()

        time.sleep(0.1)

def main():
    assistd_thread()

if __name__ == "__main__":
    main()
