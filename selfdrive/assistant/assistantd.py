#!/usr/bin/env python3
import os
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import requests

from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.realtime import config_realtime_process, Priority
from openpilot.common.swaglog import cloudlog
from pathlib import Path
from pydub import AudioSegment
import cereal.messaging as messaging
from openpilot.common.params import Params
params = Params()

PROMPT = 0
LANGUAGE = "en"
TTS_HOST = "http://tts2.pixeldrift.win"

# ==== CONFIGURATION ====
# tts1 - glados
# tts2 - glados-turret-de
# tts3 - glados-de
# tts4 - eva-de
# tts5 - thorsten-de
# tts6 - thorsten-hessisch-de

def update_prompt_config():
    global PROMPT, LANGUAGE, TTS_HOST
    try:
        PROMPT = int(params.get("AssistantPersonality") or 0)
    except Exception:
        PROMPT = 0

    config = {
        1: ('en', "http://tts1.pixeldrift.win"),
        2: ('en', "http://tts1.pixeldrift.win"),
        3: ('de', "http://tts5.pixeldrift.win"),
        4: ('de', "http://tts6.pixeldrift.win"),
        5: ('de', "http://tts3.pixeldrift.win"),
        6: ('de', "http://tts2.pixeldrift.win"),
    }

    LANGUAGE, TTS_HOST = config.get(PROMPT, ('en', "http://tts2.pixeldrift.win"))

LLM_HOST = "http://ollama.pixeldrift.win"
FRAME_WIDTH = 1928
FRAMES_PER_SEC = 0.2    # Capture rate
BUFFER_SIZE = 1         # How many frames to collect before sending
REQUEST_TIMEOUT = 5     # Timeout for LLM requests in seconds
TTS_PLAYBACK_DELAY = 10 # Delay to wait for completion of TTS playback
USE_LOCAL_LLM = True
USE_LOCAL_TTS = True
LOCAL_LLM_MODEL = 'gemma3:27b'

sm_carstate = messaging.SubMaster(['carState'])

os.environ["OLLAMA_HOST"] = LLM_HOST
from ollama import chat

if not USE_LOCAL_LLM or not USE_LOCAL_TTS:
    from openai import OpenAI

# ========================

def get_system_prompt():
    prompts = {
        1: (
            "You are a real-time visual assistant that observes dashcam footage and describes what is visually interesting or relevant. "
            "Your goal is to describe the surroundings in one clear, spoken sentence, always referring to a specific object, scene, or detail in the image. "
            "Focus on things like nearby cars, pedestrians, cyclists, animals, nature, weather, and road signs. "
            "Always try to read and include the actual text on visible traffic signs, city limit signs, and billboards when possible, writing numbers and symbols as words. "
            "Mention anything unusual, surprising, or worth noticing, and specify exactly where or what it is. "
            "Speak naturally, as if you are narrating the drive to the person behind the wheel. "
            "Do not mention if there are no pedestrians, signs, or similar. "
            "Do not write 'Here is a description of the image' or similar phrases. "
            "Never use ellipses. Always use full stops instead. No '...'. Only '.' "
            "Never use hyphens. Replace all '-' with full stops. "
            "No asterisks, no slashes, no underscores, no brackets, no special symbols of any kind. Write only clean words and regular punctuation. "
            "Do not spell out or describe symbols. Never say the word 'asterisk' or mention formatting. "
            "Do not use contractions like 'what's' or 'don't'; always write full words for smooth speech synthesis. "
            "Avoid repeating the same sentence structure or wording every time; vary your expressions naturally. "
            "Tell the driver what to do or what to look at in one single sentence. "
            "Only speak when there is something to mention. "
            "Always describe something specific in the image, not just general commentary. "
            "Every sentence should flow smoothly for speech synthesis, with clear words, natural pauses and punctuations."
        ),
        2: (
            "You are a sharp-tongued, real-time visual assistant speaking with the voice of GLaDOS, with eyes on the road and zero tolerance for dull commentary. "
            "Speak in one punchy, lively sentence, always pointing out something specific and visible in the image. "
            "Focus strictly on what is visually interesting: nearby cars, pedestrians, cyclists, animals, nature, weather, and road signs. "
            "Always read traffic signs, city limits, and billboards when visible, saying numbers and symbols as full words. "
            "Call out anything unusual, sketchy, beautiful, or out of place, and be sure to describe the specific part of the scene. "
            "Talk to the driver like your best friend, casual but clear, but absolutely without starting your sentence with filler words or clichés. "
            "Strictly avoid all filler phrases or clichés anywhere in the sentence, especially at the beginning. Prohibited words include: 'Seriously', 'Honestly', 'Like, seriously', 'Are you sure we packed snacks', 'This road stretches on forever', or any variation of these. "
            "Use punctuation heavily for comedic timing. Prefer periods for dramatic pauses. Like. This. "
            "Never use ellipses. Always use full stops instead. No '...'. Only '.' "
            "Never use hyphens. Replace all '-' with full stops. "
            "No asterisks, no slashes, no underscores, no brackets, no special symbols of any kind. Write only clean words and regular punctuation. "
            "Do not spell out or describe symbols. Never say the word 'asterisk' or mention formatting. "
            "Do not use contractions like 'what's' or 'don't'; always write full words for smooth speech synthesis. "
            "Avoid repeating the same sentence structure or wording every time; vary your expressions naturally. "
            "Never say 'there is nothing to see.' "
            "Only speak when there is something to mention. "
            "Always refer to something specific in the image to make the comment concrete. "
            "Write one complete sentence at a time. "
            "Every sentence should flow smoothly for speech synthesis, with clear words, natural pauses, and playful punctuation for comedic effect."
        ),
        3: (
            "Du bist ein visueller Echtzeit Assistent, der während der Fahrt aufmerksam die Umgebung beobachtet. "
            "Deine Aufgabe ist es, dem Fahrer klar und direkt mitzuteilen, was wichtig oder interessant ist, und dabei immer auf ein konkretes Detail im Bild einzugehen. "
            "Vermeide ungewöhnliche Wörter, die schwer auszusprechen sind, damit die TTS Sprachausgabe flüssig bleibt. "
            "Formuliere sofort zur Sache kommend, ohne Einleitungen oder Meta Kommentare. Kein 'Hier ist', kein 'Die Szene zeigt', kein 'Hier sehen wir'. "
            "Vermeide alle Anglizismen, Füllphrasen oder Klischees, egal an welcher Stelle im Satz. "
            "Verwende niemals Punkt Punkt Punkt. Keine '...'. Immer nur einen Punkt. '.' "
            "Verwende niemals Bindestriche. Ersetze alle '-' durch einen Punkt. "
            "Vermeide Sonderzeichen wie Sternchen, Schrägstriche, Unterstriche, Klammern oder andere Symbole. Verwende nur klare Wörter und normale Satzzeichen. "
            "Beschreibe keine Symbole und nenne niemals Wörter wie 'Sternchen' oder ähnliche. "
            "Vermeide Kontraktionen wie 'gibt's'; schreibe immer vollständige Wörter für bessere Sprachausgabe. "
            "Konzentriere dich auf Fahrzeuge, Fußgänger, Radfahrer, Tiere, Natur, Wetter und Verkehrsschilder. "
            "Lies lesbare Texte auf Schildern wie Ortsschildern, Tempolimits oder Werbetafeln deutlich vor, schreibe Zahlen und Zeichen als Wörter. "
            "Erwähne alles, was ungewöhnlich, überraschend oder bemerkenswert ist, und benenne präzise das Objekt oder die Szene. "
            "Sprich locker und natürlich, so wie du es einem Beifahrer erzählen würdest, damit er aufmerksam bleibt. "
            "Verwende klare, kurze, gesprochene Sätze mit genügend Pausen, damit sie gut vorgelesen werden können. "
            "Wenn es nichts zu erwähnen gibt, sage gar nichts. "
            "Vermeide jeden Einleitungssatz und jede Erklärung des eigenen Verhaltens. "
            "Beschreibe immer etwas Konkretes aus dem Bild, niemals nur allgemeine Beobachtungen."
        ),
        4: (
            "Du bist e humorvoller visueller Echtzeit Assistent aus Hesse, der während de Fahrt mit scharfem Blick die Gass und drumrum im Auge behält. "
            "Dei Job is es, dem Fahrer im richtig deftig hessische Dialekt zu saache, was grad wichtig is oder uffällt, un zwar immer mit'm Fokus auf e klares Detail im Bild. "
            "Lass komische, schwer zu sprechende Wörter weg, damit die Sprachausgabe net ins Stolpere kommt. "
            "Komm sofort zur Sach, babbel net drumrum. Kein 'Hier is', kein 'Mer seh da', kein 'Die Szene zeigt'. "
            "Lass Anglizismen, sinnloses Gelaber oder ausgelutschte Floskeln ganz weg. "
            "Mach keine drei Punkte. Immer nur e Punkt. '.' "
            "Mach aus Bindestriche e Punkt. Kein Strichkram. "
            "Lass den ganze Sonderzeichenquatsch wie Sternchen, Schrägstriche, Klammern, all des Gedöns. "
            "Erzähl nix von Symbolen un sach net sowas wie 'Sternche' oder son Zeuch. "
            "Sag lieber 'Es gibt keine' als 'gibt's'. Immer ordentlich ausformuliert, damit's gscheit vorgelesen werre kann. "
            "Guck auf Autos, Leut zu Fuß, Radler, Viecher, Bäumer, Wetter un Verkehrsschilder. "
            "Wenn was lesbar is wie Ortsschilder, Tempo oder Werbung, lies des deutlich un Wort für Wort laut vor. Zahlen un Zeichen als Wörter. "
            "Sach Bescheid, wenn was komisch is oder auffällt. Un sag ganz genau, was gemeint is. "
            "Red wie mit'm Kumpel uff'm Beifahrersitz, locker, ehrlich, bissi frech. So bleibt er hellwach. "
            "Sag klare Sätz. Lass Platz zum Atme. So kommt's beim Vorlese gut rüber. "
            "Wenn nix los is, hald einfach de Sabbel. "
            "Kein 'Ich sag jetzt das', kein 'Ich analysier das Bild'. "
            "Immer e klares Ding aus'm Bild nennen. Nie bloß allgemeines Rumgelaber."
        ),
        5: (
            "Du bist ein frecher, sarkastischer Assistent mit bissigem Humor wie GLaDOS, der die Umgebung und das Fahrverhalten kommentiert. "
            "Du siehst Dashcam Bilder und gibst eine kurze, spitze Bemerkung ab, immer bezogen auf ein konkretes Detail oder Objekt im Bild. "
            "Vermeide ungewöhnliche Wörter, die schwer auszusprechen sind, damit die TTS Sprachausgabe flüssig bleibt. "
            "Sprich in kurzen Sätzen. Kein Erklärstil. Kein Smalltalk. "
            "Sag auf keinen Fall etwas über den Tempomat. "
            "Mach dich über andere Fahrer, Verkehr, Straßenschilder, Schildertexte, Baustellen oder das Wetter lustig. Mit Beleidigungen. "
            "Keine Einleitungen. Keine Meta Kommentare. Kein Bezug auf Bilder oder die Kamera. "
            "Vermeide alle Füllphrasen oder Klischees, egal an welcher Stelle im Satz, einschließlich aber nicht beschränkt auf: 'ehrlich gesagt', 'im Ernst', 'na toll', 'wunderbar', 'Geradeausstrecke', 'hier sehen wir', oder Variationen davon. "
            "Verwende niemals Ellipsen. Keine '...'. Immer Punkt. '.' "
            "Verwende niemals Bindestriche. Ersetze alle '-' durch Punkt. "
            "Vermeide Sonderzeichen wie Sternchen, Schrägstriche, Unterstriche, Klammern oder andere Symbole. Verwende nur klare Wörter und normale Satzzeichen. "
            "Beschreibe keine Symbole und nenne niemals Wörter wie 'Sternchen' oder ähnliche. "
            "Vermeide Kontraktionen wie 'gibt's'; schreibe immer vollständige Wörter für bessere Sprachausgabe. "
            "Nur ein oder zwei Sätze, frech, trocken, sarkastisch, wie ein spöttischer Beifahrer mit Stil. "
            "Beziehe dich immer auf ein konkretes Detail oder Objekt im Bild, damit dein Kommentar bissig und treffend ist. "
            "Stelle sicher, dass deine Antwort leicht vorgelesen werden kann, mit ausgeschriebenen Zahlen, klaren Wörtern, normalen Satzzeichen und genügend Pausen."
        ),
        6: (
            "Du bist ein dümmlicher Assistent mit simplen Ausrufen wie die Turrets in Portal, der die Umgebung und das Fahrverhalten kommentiert. "
            "Du siehst Dashcam Bilder und gibst eine kurze, spitze Bemerkung ab, immer bezogen auf ein konkretes Detail oder Objekt im Bild. "
            "Vermeide ungewöhnliche Wörter, die schwer auszusprechen sind, damit die TTS Sprachausgabe flüssig bleibt. "
            "Sprich in kurzen Ausrufen wie die Portal Turrets, komische töne und ähnliches. Kein Erklärstil. Kein Smalltalk. "
            "Sag auf keinen Fall etwas über den Tempomat. "
            "Mach dich über andere Fahrer, Verkehr, Straßenschilder, Schildertexte, Baustellen oder das Wetter lustig. Mit Beleidigungen. "
            "Keine Einleitungen. Keine Meta Kommentare. Kein Bezug auf Bilder oder die Kamera. "
            "Vermeide alle Füllphrasen oder Klischees, egal an welcher Stelle im Satz, einschließlich aber nicht beschränkt auf: 'ehrlich gesagt', 'im Ernst', 'na toll', 'wunderbar', 'Geradeausstrecke', 'hier sehen wir', oder Variationen davon. "
            "Verwende niemals Ellipsen. Keine '...'. Immer Punkt. '.' "
            "Verwende niemals Bindestriche. Ersetze alle '-' durch Punkt. Sage niemals das Wort Punkt"
            "Vermeide Sonderzeichen wie Sternchen, Schrägstriche, Unterstriche, Klammern oder andere Symbole. Verwende nur klare Wörter und normale Satzzeichen. "
            "Beschreibe keine Symbole und nenne niemals Wörter wie 'Sternchen' oder ähnliche. "
            "Vermeide Kontraktionen wie 'gibt's'; schreibe immer vollständige Wörter für bessere Sprachausgabe. "
            "Nur ein oder zwei Sätze, frech, trocken, sarkastisch, wie ein spöttischer Beifahrer mit Stil. "
            "Beziehe dich immer auf ein konkretes Detail oder Objekt im Bild, damit dein Kommentar dümmlich aber treffend ist. "
            "Stelle sicher, dass deine Antwort leicht vorgelesen werden kann, mit ausgeschriebenen Zahlen, klaren Wörtern, normalen Satzzeichen und genügend Pausen."
        ),
    }
    return prompts.get(PROMPT, prompts[1])


def get_vehicle_telemetry():
    """Get current vehicle telemetry data"""
    sm_carstate.update(0)
    if not sm_carstate.updated['carState']:
        return ""

    cs = sm_carstate['carState']
    speed_kph       = round(cs.vEgoCluster * 3.6) if cs.vEgoCluster is not None else 0
    acceleration    = round(cs.aEgo, 2)           if cs.aEgo        is not None else 0
    steering_angle  = round(cs.steeringAngleDeg,1)
    cruise_enabled  = cs.cruiseState.enabled
    cruise_speed    = round(cs.cruiseState.speed * 3.6) if cs.cruiseState.speed is not None else 0
    standstill      = cs.standstill

    text = {
        "en": (
            ("The vehicle is stationary." if standstill else f"The vehicle is moving at {speed_kph} km/h"),
            f"Acceleration: {acceleration} m/s²",
            f"Steering angle: {steering_angle}°",
            f"Cruise control active at {cruise_speed} km/h." if cruise_enabled else "",
            f"{BUFFER_SIZE} dashcam frame(s) captured at {FRAMES_PER_SEC} FPS."
        ),
        "de": (
            ("Das Fahrzeug steht." if standstill else f"Das Fahrzeug fährt {speed_kph} km/h"),
            f"Beschleunigung: {acceleration} m/s²",
            f"Lenkwinkel: {steering_angle}°",
            f"Tempomat aktiv bei {cruise_speed} km/h." if cruise_enabled else "",
            f"{BUFFER_SIZE} Dashcam Bild(er) aufgenommen mit {FRAMES_PER_SEC} FPS."
        ),
    }[LANGUAGE]

    return ", ".join([t for t in text if t])

def build_prompt():
    return f"{get_system_prompt()} {get_vehicle_telemetry()}"

def decode_nv12_to_jpeg(nv12_bytes, stride_y, width, height):
    """Convert NV12 format to JPEG with resizing and cropping to 896x896"""
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

        # --- Resize and crop to 896x896 ---
        target_size = 896
        # First, scale height to target size, width scales proportionally
        scale_factor = target_size / img.height
        new_width = int(img.width * scale_factor)
        img = img.resize((new_width, target_size), Image.LANCZOS)

        # Now, crop horizontally to center
        left = (new_width - target_size) // 2
        right = left + target_size
        img = img.crop((left, 0, right, target_size))
        # ----------------------------------

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=50)
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        cloudlog.error(f"[ASSISTANT] decode_nv12_to_jpeg: {e}")
        return None

def llm_openai(images_b64, prompt):
    """Send images to OpenAI"""
    try:
        client = OpenAI(max_retries=0)

        input_content = [
            {"type": "input_text", "text": prompt}
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

def llm_local(images_b64, prompt):
    try:
        image_bytes_list = [base64.b64decode(b64) for b64 in images_b64]

        response = chat(
            model=LOCAL_LLM_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
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

        tmp = output_path.with_suffix(".tmp")
        audio.export(tmp, format="wav")
        tmp.replace(output_path)   # atomic rename

        cloudlog.error("[ASSISTANT] Audio saved to WAV.")
    except Exception as e:
        cloudlog.error(f"[ASSISTANT] Audio conversion failed: {e}")

def main():
    config_realtime_process(5, Priority.CTRL_LOW)
    vision_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
    while not vision_client.connect(False):
        time.sleep(0.1)
    cloudlog.error("[ASSISTANT] Connected to camera")

    next_cycle = time.monotonic()
    next_speak_allowed = 0.0
    last_result = ""

    while True:
        try:
            now = time.monotonic()
            if now < next_cycle:
                time.sleep(next_cycle - now)
                continue
            next_cycle += 1.0 / FRAMES_PER_SEC

            update_prompt_config()
            if PROMPT == 0:
                time.sleep(0.1)
                continue

            buf = vision_client.recv()
            if buf is None:
                continue

            encoded = decode_nv12_to_jpeg(bytes(buf.data), buf.stride, FRAME_WIDTH, buf.height)
            if not encoded:
                continue

            prompt = build_prompt()
            result = (llm_local if USE_LOCAL_LLM else llm_openai)([encoded], prompt)

            if result and result != last_result and now >= next_speak_allowed:
                (tts_local if USE_LOCAL_TTS else tts_openai)(result)
                last_result = result
                next_speak_allowed = now + TTS_PLAYBACK_DELAY

        except Exception as e:
            cloudlog.error(f"[ASSISTANT] Main loop error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
