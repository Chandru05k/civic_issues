from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import requests
import io
import shutil
import os

app = FastAPI()
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message": "YOLO model API is running! Use /predict/ for inference."}

@app.post("/predict/")
async def predict(
    file: UploadFile = File(None),
    url: str = Form(None)
):
    image = None

    # Case 1: Image file upload
    if file:
        image = Image.open(file.file).convert("RGB")

    # Case 2: Image URL
    elif url:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

    else:
        return JSONResponse(
            {"error": "No image file or URL provided."}, status_code=400
        )

    # Run YOLO
    results = model.predict(image, conf=0.25)

    detections = []
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            detections.append({
                "class": model.names[int(cls)],
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2]
            })

    return {"detections": detections}
