import requests
import base64
from PIL import Image
from io import BytesIO

SERVER_URL = "http://0.0.0.0:8000/predict"

def send_image(img_path):
    with open(img_path, "rb") as f:
        files = {"file": (img_path, f, "image/jpeg")}
        res = requests.post(SERVER_URL, files=files)
    res.raise_for_status()
    data = res.json()
    print("Predictions:", data["predictions"])

    if "annotated_base64" in data:
        img = Image.open(BytesIO(base64.b64decode(data["annotated_base64"])))
        img.save("annotated_result.jpg")
        print("Saved annotated_result.jpg")

if __name__ == "__main__":
    send_image("image.jpg")
