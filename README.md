# python_group_project
Python project for breast cancer image classification app

# Problem Statement

Breast cancer diagnosis from ultrasound images is a critical but time-consuming task. This project aims to:

Automate classification of ultrasound images
Reduce manual effort
Provide fast and consistent predictions

# Dataset Description
Input: Breast ultrasound images
Classes:
Benign
Malignant
Segmentation masks:
Automatically excluded during loading

Example structure:

training_set/
├── benign/
├── malignant/
🚀 How to Run
1. Install dependencies
pip install tensorflow numpy pytest
2. Run the program
python3 models/main.py
3. Output
Trained model saved to:
artifacts/disease_model.keras
🧠 Model Details
Input shape: 224 × 224 × 3
CNN architecture:
Conv2D + MaxPooling layers
Dense + Dropout
Output:
Binary classification (sigmoid)
Loss: Binary Crossentropy

# Requirements
Python 3.12+
TensorFlow
NumPy
pytest

Install dependencies:

# Notes(Important)
.venv/ and artifacts/ are excluded via .gitignore
Ensure dataset path is correctly set in main.py
TensorFlow must be installed in the active environment

The system:

Loads and filters ultrasound images (excluding segmentation masks)
Preprocesses and splits the dataset
Trains a deep learning model using TensorFlow
Evaluates performance
Saves the trained model for later inference
Predicts new images using a saved model

# Dataset Loader

File: models/dataset_loader.py

Handles:

Dataset validation
Filtering out mask images (files containing "mask")
Image loading and resizing
Train / validation / test splitting (70 / 15 / 15)
Normalization and batching

Key feature:

Ensures only valid ultrasound images are used for training

# Model Trainer

File: models/trainer.py

Handles:

CNN model creation
Training with early stopping
Evaluation (loss + accuracy)
Saving the trained model

Architecture:

Multiple Conv2D + MaxPooling layers
Dense + Dropout layers
Binary or multi-class support

# Predictor

File: models/predictor.py

Handles:

Loading saved model and class names
Preprocessing a single image
Predicting classification output

Returns:

predicted class
confidence
probabilities per class

# Main Pipeline

File: models/main.py

Runs the full pipeline:

loader = DatasetLoader(...)
train_ds, val_ds, test_ds = loader.build_datasets()

trainer = DiseaseModelTrainer(...)
trainer.build_model()
trainer.train(train_ds, val_ds)
trainer.evaluate(test_ds)
trainer.save_model(...)

# Tests

Folder: tests/

Contains pytest tests for:

dataset loading
model training

Dataset validation tests
Model training tests
