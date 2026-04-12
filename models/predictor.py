"""
predictor.py

Loads a saved disease-classification model and predicts the class of a
single ultrasound image.
"""

from pathlib import Path
from typing import Any
import json

import numpy as np
import tensorflow as tf


class DiseasePredictor:
    """Loads a saved model and predict disease class from an image."""

    def __init__(
        self,
        model_path: str,
        class_names_path: str,
        image_size: tuple[int, int],
    ) -> None:
        """
        Initializes the predictor.

        Args:
            model_path: Path to the saved Keras model.
            class_names_path: Path to a JSON or TXT file containing class names.
            image_size: Target image size as (height, width).
        """
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.image_size = image_size
        self.model: Any = None
        self.class_names: list[str] = []

    def load_resources(self) -> None:
        """
        Loads the saved model and class names.

        Raises:
            FileNotFoundError: If the model or class names file does not exist.
            ValueError: If no class names are found.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not self.class_names_path.exists():
            raise FileNotFoundError(
                f"Class names file not found: {self.class_names_path}"
            )

        self.model = tf.keras.models.load_model(self.model_path)

        if self.class_names_path.suffix.lower() == ".json":
            with self.class_names_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)

            if isinstance(loaded, list):
                self.class_names = [str(name) for name in loaded]
            else:
                raise ValueError(
                    "JSON class names file must contain a list of class names."
                )
        else:
            with self.class_names_path.open("r", encoding="utf-8") as file:
                self.class_names = [
                    line.strip() for line in file if line.strip()
                ]

        if not self.class_names:
            raise ValueError("No class names were loaded.")

    def preprocess_image(self, image_path: str) -> Any:
        """
        Loads, resize, normalize, and batch a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            A model-ready image tensor of shape (1, height, width, 3).

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")

        image_bytes = tf.io.read_file(str(image_file))
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, axis=0)

        return image

    def predict_image(self, image_path: str) -> dict[str, Any]:
        """
        Predicts the class of a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            A dictionary containing the predicted class, confidence,
            predicted index, and class probabilities.

        Raises:
            ValueError: If the model or class names have not been loaded.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_resources() first.")

        if not self.class_names:
            raise ValueError(
                "Class names are not loaded. Call load_resources() first."
            )

        image = self.preprocess_image(image_path)
        raw_prediction = self.model.predict(image, verbose=0)
        probabilities = raw_prediction[0]

        if self.model.output_shape[-1] == 1:
            confidence = float(probabilities[0])
            predicted_index = 1 if confidence >= 0.5 else 0
            predicted_class = self.class_names[predicted_index]

            class_probabilities = {
                self.class_names[0]: float(1.0 - confidence),
                self.class_names[1]: float(confidence),
            }

            predicted_confidence = class_probabilities[predicted_class]
        else:
            predicted_index = int(np.argmax(probabilities))
            predicted_class = self.class_names[predicted_index]
            predicted_confidence = float(probabilities[predicted_index])

            class_probabilities = {
                class_name: float(probabilities[index])
                for index, class_name in enumerate(self.class_names)
            }

        return {
            "image_path": str(image_path),
            "predicted_class": predicted_class,
            "confidence": predicted_confidence,
            "predicted_index": predicted_index,
            "class_probabilities": class_probabilities,
        }