import os
import uuid
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from models.predictor import DiseasePredictor

app = FastAPI(title="Breast Cancer Ultrasound Classifier", version="1.0.0")

# Paths can be overridden with environment variables.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(PROJECT_ROOT, "artifacts", "disease_model.keras"),
)

CLASS_NAMES_PATH = os.getenv(
    "CLASS_NAMES_PATH",
    os.path.join(PROJECT_ROOT, "artifacts", "class_names.json"),
)

# Match the image size your model was trained with.
def parse_image_size(value: str) -> tuple[int, int]:
    parts = value.replace("x", ",").split(",")
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) == 1:
        size = int(parts[0])
        return (size, size)

    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))

    raise ValueError("IMAGE_SIZE must be like '224' or '224,224'.")


IMAGE_SIZE = parse_image_size(os.getenv("IMAGE_SIZE", "224,224"))

predictor = DiseasePredictor(
    model_path=MODEL_PATH,
    class_names_path=CLASS_NAMES_PATH,
    image_size=IMAGE_SIZE,
)

predictor.load_resources()


@app.get("/", response_class=HTMLResponse)
def homepage():
    return "<h1>Breast Cancer Ultrasound Classifier</h1>"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG.")

    suffix = os.path.splitext(file.filename or "")[-1] or ".jpg"
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{suffix}")

    try:
        contents = await file.read()

        with open(tmp_path, "wb") as f:
            f.write(contents)

        result = predictor.predict_image(tmp_path)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse(
        content={
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "class_probabilities": result["class_probabilities"],
        }
    )