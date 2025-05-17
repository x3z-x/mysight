import os, threading, time
from final import start_vision
from voice import start_voice
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import ipc 
load_dotenv()

sp_key = os.getenv("SPEECH_KEY")
speech_config = speechsdk.SpeechConfig(subscription=sp_key, region=os.getenv("SPEECH_REGION", "westus"))
speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
shared_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
audio_out = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)


if __name__ == "__main__":
    t1 = threading.Thread(target=start_voice,args=(shared_synthesizer,), daemon=True)
    t2 = threading.Thread(target=start_vision, args=(shared_synthesizer,), daemon=True)

    t1.start(); t2.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down…")
