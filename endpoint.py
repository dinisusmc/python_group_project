import os
import uuid
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from predictor import DiseasePredictor

app = FastAPI(title="Breast Cancer Ultrasound Classifier", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/disease_model.keras")
predictor = BreastCancerPredictor(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
def homepage():
    return "<h1>Breast Cancer Ultrasound Classifier</h1>"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG.")

    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{suffix}")

    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)
        result = predictor.predict(tmp_path)
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
