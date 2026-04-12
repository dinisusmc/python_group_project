"""
trainer.py

Build, train, evaluate, and save a CNN model for ultrasound disease
classification.
"""

from pathlib import Path
from typing import Any

import tensorflow as tf


class DiseaseModelTrainer:
    """Builds, trains, evaluates, and saves the classification model."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        epochs: int,
    ) -> None:
        """
        Initialize thes trainer.

        Args:
            input_shape: Model input shape as (height, width, channels).
            num_classes: Number of target classes.
            epochs: Number of training epochs.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.model: Any = None

    def build_model(self) -> Any:
        """
        Builds and compiles a CNN model.
        
        """
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        output_units = 1 if self.num_classes == 2 else self.num_classes
        output_activation = "sigmoid" if self.num_classes == 2 else "softmax"
        loss_function = (
            "binary_crossentropy"
            if self.num_classes == 2
            else "sparse_categorical_crossentropy"
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.input_shape),

                tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D((2, 2)),

                tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D((2, 2)),

                tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D((2, 2)),

                tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D((2, 2)),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Dense(output_units, activation=output_activation),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss_function,
            metrics=["accuracy"],
        )

        self.model = model
        return self.model

    def train(self, train_data: Any, val_data: Any) -> Any:
        """
        Trains the model.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.

        Returns:
            Keras training history object.

        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ]

        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def evaluate(self, test_data: Any) -> dict[str, float]:
        """
        Evaluate the trained model on the test dataset.

        Args:
            test_data: Test dataset.

        Returns:
            A dictionary containing test loss and test accuracy.

        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        loss, accuracy = self.model.evaluate(test_data, verbose=0)
        return {
            "loss": float(loss),
            "accuracy": float(accuracy),
        }

    def save_model(self, output_path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            output_path: Path where the model should be saved.

        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Nothing to save.")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(output_file)