from dataset_loader import DatasetLoader
from trainer import DiseaseModelTrainer

loader = DatasetLoader(dataset_path="/Users/felipe/Downloads/iuss-23-24-automatic-diagnosis-breast-cancer/training_set")
train_ds, val_ds, test_ds = loader.build_datasets()

trainer = DiseaseModelTrainer(
    input_shape=(224, 224, 3),
    num_classes=2,
    epochs=10,
)
trainer.build_model()
trainer.train(train_ds, val_ds)
results = trainer.evaluate(test_ds)
trainer.save_model("artifacts/disease_model.keras")