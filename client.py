import cv2
import time
import requests
import os
import base64
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = os.getenv("SERVER_URL", "http://13.204.79.45:8000/predict")


def speak_text(text):
    os.system(f'espeak "{text}" -s 200')


def detections_to_speech_text(detections):
    if not detections:
        return "No objects detected."

    lines = []
    for i, det in enumerate(detections, 1):
        # confidence = round(det['confidence'], 2)
        lines.append(f"{det['class_name'].capitalize()} {i} is to the {det['direction']}.")

    return ". ".join(lines) + "."


def get_directions(detections, image_width=640):
    directions = []
    left_boundary = image_width / 3
    right_boundary = 2 * image_width / 3

    for det in detections:
        x1, y1, x2, y2 = det['box']
        center_x = (x1 + x2) / 2

        if center_x < left_boundary:
            direction = "left"
        elif center_x > right_boundary:
            direction = "right"
        else:
            direction = "straight"

        directions.append({
            'class_name': det.get('class_name', 'object'),
            'direction': direction,
            'confidence': det.get('confidence', 1.0)
        })
    return directions


def send_image_bytes(img_bytes, img_name="frame.jpg"):
    files = {"file": (img_name, img_bytes, "image/jpeg")}
    res = requests.post(SERVER_URL, files=files)
    res.raise_for_status()
    data = res.json()
    directions = get_directions(data["predictions"])
    audio_text = detections_to_speech_text(directions)
    speak_text(audio_text)


def main():
    VIDEO_CAPTURE_URL = os.getenv("VIDEO_CAPTURE_URL", "http://192.168.137.202:8080/video")
    # cap = cv2.VideoCapture("people-detection.mp4")
    cap = cv2.VideoCapture()

    if not cap.isOpened():
        print("Cannot open video")
        return

    last_capture_time = 0
    capture_interval = 2
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))

        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            last_capture_time = current_time
            frame_counter += 1

            _, img_bytes = cv2.imencode('.jpg', frame)
            send_image_bytes(img_bytes.tobytes(), img_name=f"frame_{frame_counter}.jpg")

    cap.release()


if __name__ == "__main__":
    main()
