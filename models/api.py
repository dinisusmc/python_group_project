import os
import uuid
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from predictor import DiseasePredictor

app = FastAPI(title="Breast Cancer Ultrasound Classifier", version="1.0.0")

MODEL_PATH = "../artifacts/disease_model.keras"
CLASS_NAMES_PATH = "../artifacts/class_names.json"
IMAGE_SIZE = (224, 224)

predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        p = DiseasePredictor(
            model_path=MODEL_PATH,
            class_names_path=CLASS_NAMES_PATH,
            image_size=IMAGE_SIZE,
        )
        p.load_resources()
        predictor = p
    return predictor


@app.get("/", response_class=HTMLResponse)
def homepage():
    return "<h1>Breast Cancer Ultrasound Classifier</h1>"


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    suffix = os.path.splitext(image.filename)[-1].lower() or ""
    if suffix not in allowed_extensions:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image file.")

    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{suffix}")

    try:
        contents = await image.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)
        p = get_predictor()
        result = p.predict_image(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse(content={
        "predicted_class":     result["predicted_class"],
        "confidence":          result["confidence"],
        "class_probabilities": result["class_probabilities"],
    })