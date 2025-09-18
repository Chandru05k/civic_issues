from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI(title="YOLO Object Detection API")

# Load YOLO model
model = YOLO(r"https://raw.githubusercontent.com/Chandru05k/civic_issues/main/best.pt")  # Make sure best.pt is in the same folder

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run inference
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



