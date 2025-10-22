from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64
import uvicorn

app = FastAPI(title="YOLOv8 API")

MODEL_NAME = "yolov8s.pt"
CONF_THRESHOLD = 0.25
IMG_SIZE = 640
RETURN_ANNOTATED_IMAGE = False

try:
    model = YOLO(MODEL_NAME)
    model.to("cpu")
    class_names = model.names
except Exception as e:
    model = None
    load_error = str(e)


@app.get("/")
def root():
    if not model:
        return {"status": "error", "message": load_error}
    return {"status": "ok", "model": MODEL_NAME}


def draw_boxes(img, boxes, scores, classes):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
        label = f"{class_names[int(cls)]} {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 4, y1 + 4), label, fill="red", font=font)
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail=load_error)

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_np = np.array(img)
    results = model.predict(source=img_np, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, device="cpu")
    r = results[0]

    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy()

    preds = []
    for i in range(len(xyxy)):
        preds.append({
            "box": xyxy[i].tolist(),
            "confidence": float(confs[i]),
            "class_id": int(clss[i]),
            "class_name": class_names[int(clss[i])]
        })

    resp = {"predictions": preds}

    if RETURN_ANNOTATED_IMAGE and len(preds) > 0:
        annotated = draw_boxes(img.copy(), xyxy, confs, clss)
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG")
        resp["annotated_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse(resp)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
