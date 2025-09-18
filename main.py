import io
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="YOLO Object Detection API")

# ----------------------------
# Model download & setup
# ----------------------------
weights_path = Path("weights/best.pt")
weights_path.parent.mkdir(exist_ok=True)

# GitHub raw URL for the model
url = "https://github.com/Chandru05k/civic_issues/raw/main/best.pt"

# Download model if it doesn't exist or is empty
if not weights_path.exists() or weights_path.stat().st_size == 0:
    print("Downloading YOLO model...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(weights_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded successfully.")

# Load YOLO model
model = YOLO(str(weights_path))
print("YOLO model loaded successfully.")

# ----------------------------
# API endpoint
# ----------------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes and convert to RGB
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Run prediction
        results = model.predict(image_np, conf=0.25, imgsz=640)

        # Parse detections
        detections = []
        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                detections.append({
                    "class": model.names[int(cls)],
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2]
                })

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
