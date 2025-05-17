import os
import azure.cognitiveservices.speech as speechsdk
import google.generativeai as genai
from dotenv import load_dotenv
from ipc import set_flag, clear_flag

load_dotenv()
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')  # Google Gemini API Key
AZURE_SPEECH_KEY = os.getenv('SPEECH_KEY')    # Azure Speech Service API Key
AZURE_SPEECH_REGION = os.getenv('SPEECH_REGION')  # Azure Speech Service Region
if not GEMINI_API_KEY:
    raise ValueError("Please set your GOOGLE_API_KEY environment variable.")
if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
    raise ValueError("Please set your SPEECH_KEY and SPEECH_REGION environment variables.")

genai.configure(api_key=GEMINI_API_KEY)


def listen_for_wake_word(shared_synthesizer: speechsdk.SpeechSynthesizer, wake_word="hi my sign") -> bool:

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Listening for the wake word...")

    while True:
        print("Say something...")
        result = speech_recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            phrase = result.text.lower()
            print(f"You said: {phrase}")
            if wake_word in phrase:
                set_flag()
                speak_response(shared_synthesizer, "Hi, how can I assist?")
                print("Wake word detected!")
                return True
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            print(f"Recognition canceled: {details.reason}")
            if details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {details.error_details}")


def capture_question() -> str | None:
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Listening for your question...")

    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"You asked: {result.text}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print(f"Recognition canceled: {details.reason}")
        if details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {details.error_details}")
    return None


def get_ai_response(prompt: str) -> str | None:
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=150,
            )
        )
        text = resp.text or "Sorry, I couldn't generate a response."
        print(f"AI Assistant: {text}")
        return text
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return None


def speak_response(speech_synthesizer, response_text: str) -> None:
    result = speech_synthesizer.speak_text_async(response_text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for the response.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        print(f"Speech synthesis canceled: {details.reason}")
        if details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {details.error_details}")
    clear_flag()


def start_voice(shared_synthesizer: speechsdk.SpeechSynthesizer) -> None:
    while True:
        if listen_for_wake_word(shared_synthesizer):
            question = capture_question()
            if question:
                ai_response = get_ai_response(question)
                if ai_response:
                    speak_response(shared_synthesizer, ai_response)

