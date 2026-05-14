"""

Run from the project root:
    pytest tests/test_trainer.py -v
"""

import importlib
import inspect

import pytest


tf = pytest.importorskip("tensorflow")


@pytest.fixture(scope="module")
def trainer_module():
    """Import the trainer module from the project."""
    try:
        return importlib.import_module("models.trainer")
    except ModuleNotFoundError:
        return importlib.import_module("trainer")


def _find_trainer_class(module):
    """Find a likely trainer class in trainer.py."""
    for name in ("Trainer", "ModelTrainer", "DiseaseTrainer", "BreastCancerTrainer"):
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _make_trainer(TrainerClass, **kwargs):
    """Instantiate trainer while tolerating constructor differences."""
    signature = inspect.signature(TrainerClass)
    accepted_kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return TrainerClass(**accepted_kwargs)


def _build_model_from_module_or_class(trainer_module, num_classes=3, image_size=(64, 64)):
    """Build a model using the trainer's public API."""
    TrainerClass = _find_trainer_class(trainer_module)

    if TrainerClass is not None:
        trainer = _make_trainer(
            TrainerClass,
            image_size=image_size,
            input_shape=(*image_size, 3),
            num_classes=num_classes,
            learning_rate=0.001,
        )

        for method_name in ("build_model", "create_model", "get_model"):
            if hasattr(trainer, method_name):
                method = getattr(trainer, method_name)
                try:
                    return method(num_classes=num_classes)
                except TypeError:
                    try:
                        return method(input_shape=(*image_size, 3), num_classes=num_classes)
                    except TypeError:
                        return method()

    for function_name in ("build_model", "create_model", "get_model"):
        if hasattr(trainer_module, function_name):
            function = getattr(trainer_module, function_name)
            try:
                return function(input_shape=(*image_size, 3), num_classes=num_classes)
            except TypeError:
                try:
                    return function(num_classes=num_classes)
                except TypeError:
                    return function()

    pytest.skip("trainer.py does not expose a model-building class or function.")


def test_build_model_returns_compiled_keras_model(trainer_module):
    """The trainer should build a real compiled Keras model, not a placeholder."""
    model = _build_model_from_module_or_class(trainer_module, num_classes=3)

    assert isinstance(model, tf.keras.Model)
    assert model.optimizer is not None, "Model should be compiled with an optimizer."
    assert model.loss is not None, "Model should be compiled with a loss function."


def test_model_output_matches_number_of_classes(trainer_module):
    """The final model output should match the number of disease classes."""
    num_classes = 3
    model = _build_model_from_module_or_class(trainer_module, num_classes=num_classes)

    assert model.output_shape[-1] == num_classes


def test_model_uses_global_average_pooling_instead_of_flatten(trainer_module):
    """The revised CNN should use GlobalAveragePooling2D instead of Flatten."""
    model = _build_model_from_module_or_class(trainer_module, num_classes=3)
    layer_types = {type(layer).__name__ for layer in model.layers}

    assert "GlobalAveragePooling2D" in layer_types
    assert "Flatten" not in layer_types


def test_model_contains_batch_normalization_layers(trainer_module):
    """The convolution blocks should include BatchNormalization layers."""
    model = _build_model_from_module_or_class(trainer_module, num_classes=3)
    batch_norm_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]

    assert batch_norm_layers, "Expected at least one BatchNormalization layer."


def test_model_tracks_precision_recall_and_auc_metrics(trainer_module):
    """The compiled model should track more than accuracy."""
    model = _build_model_from_module_or_class(trainer_module, num_classes=3)
    metric_names = {metric.name.lower() for metric in model.metrics}

    joined_metric_names = " ".join(metric_names)
    assert "precision" in joined_metric_names
    assert "recall" in joined_metric_names
    assert "auc" in joined_metric_names


def test_trainer_defines_reduce_lr_and_early_stopping_callbacks(trainer_module):
    """Trainer should provide EarlyStopping and ReduceLROnPlateau callbacks."""
    callback_output = None

    TrainerClass = _find_trainer_class(trainer_module)
    if TrainerClass is not None:
        trainer = _make_trainer(TrainerClass, learning_rate=0.001)
        for method_name in ("get_callbacks", "create_callbacks", "build_callbacks"):
            if hasattr(trainer, method_name):
                callback_output = getattr(trainer, method_name)()
                break

    if callback_output is None:
        for function_name in ("get_callbacks", "create_callbacks", "build_callbacks"):
            if hasattr(trainer_module, function_name):
                callback_output = getattr(trainer_module, function_name)()
                break

    if callback_output is None:
        pytest.skip("trainer.py does not expose a callback-building method.")

    callback_types = {type(callback).__name__ for callback in callback_output}
    assert "EarlyStopping" in callback_types
    assert "ReduceLROnPlateau" in callback_types
