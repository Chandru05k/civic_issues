import requests
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="YOLO Object Detection API")

# Local path to save the model
weights_path = Path("weights/best.pt")
weights_path.parent.mkdir(exist_ok=True)  # Create folder if not exists

# Raw GitHub URL
url = "https://raw.githubusercontent.com/Chandru05k/civic_issues/main/best.pt"

# Download the file if it doesn't exist locally
if not weights_path.exists():
    response = requests.get(url)
    response.raise_for_status()
    weights_path.write_bytes(response.content)

# Load YOLO model from local file
model = YOLO(str(weights_path))

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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

    return JSONResponse(content={"detections": detections})
