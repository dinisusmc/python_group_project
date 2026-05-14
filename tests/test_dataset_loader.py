"""
test_dataset_loader.py

These tests intentionally create a tiny temporary image dataset so they do
not depend on your local Kaggle folder path. They verify actual behavior:
- dataset validation rejects missing folders
- class names are discovered from class subfolders
- TensorFlow datasets can be created from a valid image directory

Run from the project root:
    pytest tests/test_dataset_loader.py -v
"""

import importlib
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture(scope="module")
def dataset_loader_module():
    """Import the DatasetLoader module from the project."""
    try:
        return importlib.import_module("models.dataset_loader")
    except ModuleNotFoundError:
        return importlib.import_module("dataset_loader")


@pytest.fixture
def tiny_image_dataset(tmp_path):
    """Create a small image dataset with two classes and valid PNG files."""
    dataset_dir = tmp_path / "tiny_dataset"
    class_names = ["benign", "malignant"]

    for class_name in class_names:
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True)
        for index in range(2):
            image = Image.new("RGB", (32, 32), color=(index * 40, 20, 120))
            image.save(class_dir / f"{class_name}_{index}.png")

    return dataset_dir, class_names


def _make_loader(DatasetLoader, dataset_path, image_size=(32, 32), batch_size=2):
    """Create DatasetLoader while tolerating small constructor differences."""
    try:
        return DatasetLoader(
            dataset_path=str(dataset_path),
            image_size=image_size,
            batch_size=batch_size,
        )
    except TypeError:
        try:
            return DatasetLoader(str(dataset_path), image_size=image_size, batch_size=batch_size)
        except TypeError:
            return DatasetLoader(str(dataset_path))


def test_validate_dataset_rejects_missing_path(dataset_loader_module, tmp_path):
    """Dataset validation should fail when the dataset folder does not exist."""
    DatasetLoader = dataset_loader_module.DatasetLoader
    missing_path = tmp_path / "does_not_exist"
    loader = _make_loader(DatasetLoader, missing_path)

    if hasattr(loader, "validate_dataset"):
        with pytest.raises((FileNotFoundError, ValueError, AssertionError)):
            loader.validate_dataset()
    else:
        pytest.skip("DatasetLoader does not expose validate_dataset().")


def test_get_class_names_returns_real_subfolders(dataset_loader_module, tiny_image_dataset):
    """Class names should come from the real class folders in the dataset."""
    DatasetLoader = dataset_loader_module.DatasetLoader
    dataset_dir, expected_class_names = tiny_image_dataset
    loader = _make_loader(DatasetLoader, dataset_dir)

    if hasattr(loader, "get_class_names"):
        class_names = loader.get_class_names()
    elif hasattr(loader, "class_names"):
        class_names = loader.class_names
    else:
        pytest.skip("DatasetLoader does not expose class-name functionality.")

    assert sorted(class_names) == sorted(expected_class_names)
    assert len(class_names) == 2


def test_load_dataset_returns_tensorflow_datasets(dataset_loader_module, tiny_image_dataset):
    """Loading a valid dataset should return one or more TensorFlow dataset objects."""
    DatasetLoader = dataset_loader_module.DatasetLoader
    dataset_dir, _ = tiny_image_dataset
    loader = _make_loader(DatasetLoader, dataset_dir, image_size=(32, 32), batch_size=2)

    load_method = None
    for method_name in ("load_dataset", "load_data", "create_datasets", "get_datasets"):
        if hasattr(loader, method_name):
            load_method = getattr(loader, method_name)
            break

    if load_method is None:
        pytest.skip("DatasetLoader does not expose a dataset-loading method.")

    output = load_method()

    if isinstance(output, tuple):
        datasets = [item for item in output if hasattr(item, "take")]
    else:
        datasets = [output] if hasattr(output, "take") else []

    assert datasets, "Expected at least one TensorFlow dataset-like object."

    first_batch = next(iter(datasets[0].take(1)))
    assert len(first_batch) >= 2, "Expected image and label tensors in each batch."

    images, labels = first_batch[0], first_batch[1]
    assert images.shape[0] > 0
    assert labels.shape[0] > 0