from pathlib import Path
from typing import Any

import tensorflow as tf


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DatasetLoader:
    """
    Loads and validates an ultrasound image dataset for binary classification.

    """
    def __init__(
        self,
        dataset_path: str,
        image_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        seed: int = 123,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        self._class_names: list[str] | None = None

    def validate_dataset(self) -> None:
        """
        Validates dataset structure and ensure each class contains
        usable non-mask image files.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        if not self.dataset_path.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {self.dataset_path}")

        class_names = self.get_class_names()
        if len(class_names) < 2:
            raise ValueError(
                f"Expected at least 2 class folders, found {len(class_names)} in {self.dataset_path}"
            )

        class_files = self._collect_classification_files()
        empty_classes = [class_name for class_name, files in class_files.items() if len(files) == 0]

        if empty_classes:
            raise ValueError(
                "The following class folders does not contain usable non-mask image files: "
                + ", ".join(empty_classes)
            )

    def get_class_names(self) -> list[str]:
        """
        Returns sorted class folder names.
        """
        if self._class_names is not None:
            return self._class_names

        if not self.dataset_path.exists() or not self.dataset_path.is_dir():
            return []

        class_names = sorted(
            [
                item.name
                for item in self.dataset_path.iterdir()
                if item.is_dir() and not item.name.startswith(".")
            ]
        )

        self._class_names = class_names
        return self._class_names

    def _is_image_file(self, file_path: Path) -> bool:
        """
        Returns True only for valid image files.
        """
        return file_path.is_file() and file_path.suffix.lower() in _IMAGE_SUFFIXES

    def _is_mask_file(self, file_path: Path) -> bool:
        """
        Detects segmentation mask filenames.

        """
        filename = file_path.stem.lower()
        return "mask" in filename

    def _collect_classification_files(self) -> dict[str, list[Path]]:

        class_files: dict[str, list[Path]] = {}

        for class_name in self.get_class_names():
            class_dir = self.dataset_path / class_name

            if not class_dir.is_dir():
                class_files[class_name] = []
                continue

            files = [
                file_path
                for file_path in class_dir.iterdir()
                if self._is_image_file(file_path) and not self._is_mask_file(file_path)
            ]

            class_files[class_name] = sorted(files)

        return class_files
    

    def get_dataset_summary(self) -> dict[str, Any]:
        """
        Return helpful dataset metadata for debugging and display.
        """
        class_files = self._collect_classification_files()

        per_class_counts = {
            class_name: len(files)
            for class_name, files in class_files.items()
        }

        total_images = sum(per_class_counts.values())

        masks_detected = 0
        for class_name in self.get_class_names():
            class_dir = self.dataset_path / class_name
            if class_dir.is_dir():
                masks_detected += sum(
                    1
                    for file_path in class_dir.iterdir()
                    if self._is_image_file(file_path) and self._is_mask_file(file_path)
                )

        return {
            "dataset_path": str(self.dataset_path),
            "class_names": self.get_class_names(),
            "num_classes": len(self.get_class_names()),
            "total_images": total_images,
            "per_class_counts": per_class_counts,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "masks_detected": masks_detected,
        }

    def _normalize(self, image: Any, label: Any) -> tuple[Any, Any]:
        """
        Converts image to float32 and scale pixels to [0, 1].
        """
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def _prepare_dataset(self, dataset: Any, training: bool = False) -> Any:
        """
        Normalize, optionally shuffle, and prefetch dataset.
        """
        autotune = tf.data.AUTOTUNE

        dataset = dataset.map(self._normalize, num_parallel_calls=autotune)

        if training:
            dataset = dataset.shuffle(buffer_size=1000, seed=self.seed)

        dataset = dataset.prefetch(buffer_size=autotune)
        return dataset

    def build_datasets(self) -> tuple[Any, Any, Any]:
        """
        Build train / validation / test datasets from filtered file lists.

        Split strategy:
        - 70% train
        - 15% validation
        - 15% test
        """
        self.validate_dataset()

        class_names = self.get_class_names()
        class_files = self._collect_classification_files()

        file_paths: list[str] = []
        labels: list[int] = []

        for label_index, class_name in enumerate(class_names):
            for file_path in class_files[class_name]:
                file_paths.append(str(file_path))
                labels.append(label_index)

        if len(file_paths) < 3:
            raise ValueError(
                "Need at least 3 usable images to create train, validation, and test splits."
            )

        path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)

        def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, self.image_size)
            image.set_shape((self.image_size[0], self.image_size[1], 3))
            return image, label

        dataset = tf.data.Dataset.zip((path_ds, label_ds))
        dataset = dataset.shuffle(len(file_paths), seed=self.seed, reshuffle_each_iteration=False)
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

        total_count = len(file_paths)
        train_count = int(total_count * 0.70)
        val_count = int(total_count * 0.15)
        test_count = total_count - train_count - val_count

        if train_count == 0 or val_count == 0 or test_count == 0:
            raise ValueError(
                "Dataset is too small after filtering. "
                "Need enough usable images so train, validation, and test are all non-empty."
            )

        train_ds = dataset.take(train_count)
        remainder_ds = dataset.skip(train_count)
        val_ds = remainder_ds.take(val_count)
        test_ds = remainder_ds.skip(val_count)

        train_ds = train_ds.batch(self.batch_size)
        val_ds = val_ds.batch(self.batch_size)
        test_ds = test_ds.batch(self.batch_size)

        train_ds = self._prepare_dataset(train_ds, training=True)
        val_ds = self._prepare_dataset(val_ds, training=False)
        test_ds = self._prepare_dataset(test_ds, training=False)

        return train_ds, val_ds, test_ds