from predictor import DiseasePredictor

predictor = DiseasePredictor(
    model_path="artifacts/disease_model.keras",
    class_names_path="artifacts/class_names.json",
    image_size=(224, 224),
)

predictor.load_resources()

result = predictor.predict_image(
    "/Users/felipe/Downloads/iuss-23-24-automatic-diagnosis-breast-cancer/training_set/benign/benign (151).png"
)

print(result)