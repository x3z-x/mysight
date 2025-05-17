import cv2
import io
import socket
import struct
import pickle
import time
import os
import threading

from dotenv import load_dotenv
from queue import Queue, Empty
from picamera2 import Picamera2
from ipc import is_flag_set

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

# Configuration
subscription_key = os.getenv("API_key")
endpoint         = os.getenv("ENDPOINT")
SERVER_HOST      = '192.168.1.10'
SERVER_PORT      = 9999


def start_vision(speech_synthesizer: speechsdk.SpeechSynthesizer):
    cv_client = ComputerVisionClient(
        endpoint,
        CognitiveServicesCredentials(subscription_key)
    )

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    ))
    cam.start()
    time.sleep(2)

    frame_queue    = Queue()
    frame_interval = 20
    frame_count    = 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((SERVER_HOST, SERVER_PORT))
        print("Connected to detection server")
    except Exception as e:
        print(f"Server connection failed: {e}")
        sock = None

    def analyze_loop():
        while True:
            while is_flag_set():
                time.sleep(0.1)

            try:
                item = frame_queue.get(timeout=0.5)
            except Empty:
                continue

            if item is None:
                break

            frame_bytes, idx = item
            try:
                with io.BytesIO(frame_bytes) as img_stream:
                    analysis = cv_client.analyze_image_in_stream(
                        img_stream,
                        visual_features=["Description", "Tags", "Objects"]
                    )
                if analysis.description and analysis.description.captions:
                    for cap in analysis.description.captions:
                        print(f"Caption: {cap.text} ({cap.confidence:.2f})")
                if any(tag.name == 'text' for tag in analysis.tags):
                    with io.BytesIO(frame_bytes) as txt_stream:
                        op = cv_client.read_in_stream(txt_stream, raw=True)
                        op_id = op.headers["Operation-Location"].split('/')[-1]
                        while True:
                            res = cv_client.get_read_result(op_id)
                            if res.status not in (OperationStatusCodes.not_started, OperationStatusCodes.running):
                                break
                            time.sleep(0.5)
                        if res.status == OperationStatusCodes.succeeded:
                            for page in res.analyze_result.read_results:
                                for line in page.lines:
                                    speech_synthesizer.speak_text_async(line.text).get()
                                    print(f"Text: {line.text}")
                if analysis.objects:
                    print("Azure Objects:")
                    for obj in analysis.objects:
                        print(f" - {obj.object_property} ({obj.confidence:.2f})")

                if sock:
                    msg = struct.pack("I", len(frame_bytes)) + frame_bytes
                    sock.sendall(msg)
                    raw = sock.recv(4)
                    if len(raw) < 4:
                        continue
                    size = struct.unpack("I", raw)[0]
                    data = b""
                    while len(data) < size:
                        data += sock.recv(4096)
                    resp = pickle.loads(data[:size])
                    for det in resp.get('detection_results', []):
                        label = det['label']
                        conf  = det['confidence']
                        dist  = det['distance_cm']
                        alert = f"{label} detected {dist:.1f} cm away"
                        speech_synthesizer.speak_text_async(alert).get()
                        print(f"{alert} (Confidence: {conf:.2f})")

            except Exception as e:
                print(f"Processing error: {e}")

        if sock:
            sock.close()

    t = threading.Thread(target=analyze_loop, daemon=True)
    t.start()

    print("Press 'q' to quit")
    try:
        while True:
            while is_flag_set():
                time.sleep(0.1)

            frame = cam.capture_array()
            cv2.imshow("Camera Feed", frame)

            if frame_count % frame_interval == 0:
                _, buf = cv2.imencode('.jpg', frame)
                frame_queue.put((buf.tobytes(), frame_count))
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        frame_queue.put(None)
        t.join()
        cam.stop()
        cam.close()
        cv2.destroyAllWindows()


