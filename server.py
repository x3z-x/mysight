# server.py

import socket
import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import struct
import logging

def estimate_distance(width_in_frame, known_width, focal_length):
    return (known_width * focal_length) / width_in_frame

logging.getLogger('ultralytics').setLevel(logging.WARNING)

KNOWN_WIDTH = 0.5
FOCAL_LENGTH = 17.91496063 

model = YOLO('yolov8n.pt')

HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 9999       # Arbitrary non-privileged port

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f'Server listening on {HOST}:{PORT}')

def handle_client(conn, addr):
    print(f'Connected by {addr}')
    data = b''
    payload_size = struct.calcsize("I") 

    try:
        while True:
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    return 
                data += packet

            if len(data) < payload_size:
                return 

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("I", packed_msg_size)[0]

            while len(data) < msg_size:
                packet = conn.recv(4096)
                if not packet:
                    return 
                data += packet

            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            results = model(frame)
            result = results[0]

            detection_results = []

            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = model.names[class_id]

                width_in_frame = x2 - x1

                distance = estimate_distance(width_in_frame, KNOWN_WIDTH, FOCAL_LENGTH)
                distance = distance * 100 

                detection_results.append({
                    'label': label,
                    'confidence': confidence,
                    'distance_cm': distance,
                    'bbox': [x1, y1, x2, y2]
                })

            num_detections = len(boxes)
            height, width = result.orig_shape[:2]
            speed = result.speed 

            detection_info = f"({num_detections} detections)" if num_detections > 0 else "(no detections)"
            inference_output = (
                f"0: {height}x{width} {detection_info}, {speed['inference']:.1f}ms\n"
                f"Speed: {speed['preprocess']:.1f}ms preprocess, "
                f"{speed['inference']:.1f}ms inference, "
                f"{speed['postprocess']:.1f}ms postprocess per image at shape {result.orig_shape}"
            )

            response = {
                'detection_results': detection_results,
                'inference_output': inference_output
            }
            print(response)

            data_to_send = pickle.dumps(response)
            conn.sendall(struct.pack("I", len(data_to_send)))
            conn.sendall(data_to_send)

    except Exception as e:
        print(f"Exception with client {addr}: {e}")
    finally:
        print(f"Client {addr} disconnected")
        conn.close()

try:
    while True:
        conn, addr = server_socket.accept()
        handle_client(conn, addr)
except KeyboardInterrupt:
    print("\nServer shutting down...")
finally:
    server_socket.close()
