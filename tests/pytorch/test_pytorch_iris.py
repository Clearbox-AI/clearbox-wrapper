import os
from sys import version_info

import pytest
import yaml

import pandas as pd
import numpy as np

import sklearn.datasets as datasets
import sklearn.preprocessing as sk_preprocessing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clearbox_wrapper.clearbox_wrapper as cbw


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture(scope="module")
def iris_data():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    return x, y


def get_dataset(data):
    x, y = data
    dataset = [(xi.astype(np.float32), yi.astype(np.float32)) for xi, yi in zip(x, y)]
    return dataset


def train_model(model, data):
    dataset = get_dataset(data)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch_size = 16
    num_workers = 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    for _ in range(5):
        for batch in dataloader:
            optimizer.zero_grad()
            batch_size = batch[0].shape[0]
            y_pred = model(batch[0]).squeeze(dim=1)
            loss = criterion(y_pred, batch[1])
            loss.backward()
            optimizer.step()


@pytest.fixture(scope="module")
def sequential_model(iris_data):
    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )

    train_model(model=model, data=iris_data)
    return model


def _predict(model, data):
    dataset = get_dataset(data)
    batch_size = 16
    num_workers = 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    predictions = np.zeros((len(dataloader.sampler),))
    model.eval()
    with torch.no_grad():
        print("===== PREDICT ======")
        for i, batch in enumerate(dataloader):
            if i == 0:
                print(
                    "- batch type: {}, batch shape: {}".format(
                        type(batch[0]), batch[0].shape
                    )
                )
                print("- batch[0][0]: {}".format(batch[0][0]))
                for el in batch[0][0]:
                    print(
                        "- el type: {}, el value: {}".format(type(el.item()), el.item())
                    )
            y_preds = model(batch[0]).squeeze(dim=1).numpy()
            predictions[i * batch_size : (i + 1) * batch_size] = y_preds
    return predictions


def _loaded_predict(model, data):
    dataset = get_dataset(data)
    batch_size = 16
    num_workers = 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    predictions = np.zeros((len(dataloader.sampler),))
    with torch.no_grad():
        print("===== PREDICT LOADED ======")
        for i, batch in enumerate(dataloader):
            if i == 0:
                print(
                    "- batch type: {}, batch shape: {}".format(
                        type(batch[0]), batch[0].shape
                    )
                )
                print("- batch[0][0]: {}".format(batch[0][0]))
                for el in batch[0][0]:
                    print(
                        "- el type: {}, el value: {}".format(type(el.item()), el.item())
                    )
            y_preds = model.predict(batch[0]).squeeze(dim=1).numpy()
            predictions[i * batch_size : (i + 1) * batch_size] = y_preds
    return predictions


@pytest.fixture(scope="module")
def sequential_predicted(sequential_model, iris_data):
    return _predict(sequential_model, iris_data)


@pytest.fixture()
def sk_function_transformer():
    def simple_preprocessor(numpy_x):
        return numpy_x ** 2

    transformer = sk_preprocessing.FunctionTransformer(
        simple_preprocessor, validate=True
    )
    return transformer


@pytest.fixture()
def custom_transformer():
    def simple_preprocessor(numpy_x):
        transformed_x = numpy_x + 1.0
        return transformed_x

    return simple_preprocessor


@pytest.fixture()
def drop_column_transformer():
    def drop_column(numpy_x):
        transformed_x = np.delete(numpy_x, 0, axis=1)
        return transformed_x

    return drop_column


def test_iris_pytorch_no_preprocessing(iris_data, model_path):

    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )
    train_model(model, iris_data)
    cbw.save_model(model_path, model)

    loaded_model = cbw.load_model(model_path)

    original_model_predictions = _predict(model, iris_data)
    loaded_model_predictions = _loaded_predict(loaded_model, iris_data)

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


@pytest.mark.parametrize(
    "sk_transformer",
    [
        (sk_preprocessing.StandardScaler()),
        (sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50)),
        (sk_preprocessing.KBinsDiscretizer(n_bins=2, encode="ordinal")),
        (sk_preprocessing.RobustScaler()),
        (sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_pytorch_preprocessing(sk_transformer, iris_data, model_path):
    x, y = iris_data
    x_transformed = sk_transformer.fit_transform(x)

    print("-- X[0]: {}".format(x[0]))
    print("-- X_TRANSFORMED[0]: {}".format(x_transformed[0]))
    print("-- Y[0]: {}".format(y[0]))

    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )
    train_model(model, (x_transformed, y))
    cbw.save_model(model_path, model, sk_transformer)

    loaded_model = cbw.load_model(model_path)

    original_model_predictions = _predict(model, (x_transformed, y))
    loaded_model_predictions = _loaded_predict(loaded_model, iris_data)

    np.testing.assert_almost_equal(
        original_model_predictions, loaded_model_predictions, decimal=2
    )

    """
    model = keras_model
    model.fit(x_transformed, y)
    cbw.save_model(model_path, model, sk_transformer)

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = model.predict(x_transformed)
    loaded_model_predictions = loaded_model.predict(x)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)



def test_iris_keras_preprocessing_with_function_transformer(
    sk_function_transformer, iris_data, keras_model, model_path
):
    x, y = iris_data
    x_transformed = sk_function_transformer.fit_transform(x)

    model = keras_model
    model.fit(x_transformed, y)
    cbw.save_model(model_path, model, sk_function_transformer)

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = model.predict(x_transformed)
    loaded_model_predictions = loaded_model.predict(x)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_keras_preprocessing_with_custom_transformer(
    custom_transformer, iris_data, keras_model, model_path
):
    x, y = iris_data
    x_transformed = custom_transformer(x)

    model = keras_model
    model.fit(x_transformed, y)
    cbw.save_model(model_path, model, custom_transformer)

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = model.predict(x_transformed)
    loaded_model_predictions = loaded_model.predict(x)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


@pytest.mark.parametrize(
    "preprocessor",
    [
        (sk_preprocessing.StandardScaler()),
        (sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50)),
        (sk_preprocessing.KBinsDiscretizer(n_bins=2, encode="ordinal")),
        (sk_preprocessing.RobustScaler()),
        (sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_keras_data_cleaning_and_preprocessing(
    preprocessor, drop_column_transformer, iris_data, model_path
):
    x, y = iris_data
    x_cleaned = drop_column_transformer(x)
    x_transformed = preprocessor.fit_transform(x_cleaned)

    model = Sequential()
    model.add(Dense(8, input_dim=x_transformed.shape[1], activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(x_transformed, y)

    cbw.save_model(model_path, model, preprocessor, drop_column_transformer)

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = model.predict(x_transformed)
    loaded_model_predictions = loaded_model.predict(x)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_keras_data_cleaning_without_preprocessing(
    iris_data, keras_model, model_path
):
    x, y = iris_data
    sk_transformer = sk_preprocessing.StandardScaler()
    x_transformed = sk_transformer.fit_transform(x)

    model = keras_model
    model.fit(x_transformed, y)

    with pytest.raises(ValueError):
        cbw.save_model(model_path, model, data_cleaning=drop_column_transformer)


def test_iris_keras_load_preprocessing_without_preprocessing(
    iris_data, keras_model, model_path
):
    x, y = iris_data
    model = keras_model
    model.fit(x, y)
    cbw.save_model(model_path, model)

    with pytest.raises(FileNotFoundError):
        loaded_model, preprocessing = cbw.load_model_preprocessing(model_path)


def test_iris_keras_load_data_cleaning_without_data_cleaning(
    iris_data, keras_model, model_path
):
    x, y = iris_data
    sk_transformer = sk_preprocessing.StandardScaler()
    x_transformed = sk_transformer.fit_transform(x)

    model = keras_model
    model.fit(x_transformed, y)
    cbw.save_model(model_path, model, sk_transformer)

    with pytest.raises(FileNotFoundError):
        (
            loaded_model,
            preprocessing,
            data_cleaning,
        ) = cbw.load_model_preprocessing_data_cleaning(model_path)


@pytest.mark.parametrize(
    "preprocessor",
    [
        (sk_preprocessing.StandardScaler()),
        (sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50)),
        (sk_preprocessing.KBinsDiscretizer(n_bins=2, encode="ordinal")),
        (sk_preprocessing.RobustScaler()),
        (sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_keras_get_preprocessed_data(
    preprocessor, iris_data, keras_model, model_path
):
    x, y = iris_data
    x_transformed = preprocessor.fit_transform(x)

    model = keras_model
    model.fit(x_transformed, y)
    cbw.save_model(model_path, model, preprocessor)

    loaded_model, loaded_preprocessing = cbw.load_model_preprocessing(model_path)
    x_transformed_by_loaded_preprocessing = loaded_preprocessing(x)
    np.testing.assert_array_equal(x_transformed, x_transformed_by_loaded_preprocessing)


@pytest.mark.parametrize(
    "preprocessor",
    [
        (sk_preprocessing.StandardScaler()),
        (sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50)),
        (sk_preprocessing.KBinsDiscretizer(n_bins=2, encode="ordinal")),
        (sk_preprocessing.RobustScaler()),
        (sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_keras_get_cleaned_data(
    preprocessor, drop_column_transformer, iris_data, model_path
):
    x, y = iris_data
    x_cleaned = drop_column_transformer(x)
    x_transformed = preprocessor.fit_transform(x_cleaned)

    model = Sequential()
    model.add(Dense(8, input_dim=x_transformed.shape[1], activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(x_transformed, y)

    cbw.save_model(model_path, model, preprocessor, drop_column_transformer)

    (
        loaded_model,
        loaded_preprocessing,
        loaded_data_cleaning,
    ) = cbw.load_model_preprocessing_data_cleaning(model_path)

    x_cleaned_by_loaded_data_cleaning = loaded_data_cleaning(x)
    np.testing.assert_array_equal(x_cleaned, x_cleaned_by_loaded_data_cleaning)


@pytest.mark.parametrize(
    "preprocessor",
    [
        (sk_preprocessing.StandardScaler()),
        (sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50)),
        (sk_preprocessing.KBinsDiscretizer(n_bins=2, encode="ordinal")),
        (sk_preprocessing.RobustScaler()),
        (sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_keras_get_cleaned_and_processed_data(
    preprocessor, drop_column_transformer, iris_data, model_path
):
    x, y = iris_data
    x_cleaned = drop_column_transformer(x)
    x_transformed = preprocessor.fit_transform(x_cleaned)

    model = Sequential()
    model.add(Dense(8, input_dim=x_transformed.shape[1], activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(x_transformed, y)

    cbw.save_model(model_path, model, preprocessor, drop_column_transformer)

    (
        loaded_model,
        loaded_preprocessing,
        loaded_data_cleaning,
    ) = cbw.load_model_preprocessing_data_cleaning(model_path)

    x_cleaned_by_loaded_data_cleaning = loaded_data_cleaning(x)
    x_transformed_by_loaded_preprocessing = loaded_preprocessing(
        x_cleaned_by_loaded_data_cleaning
    )
    np.testing.assert_array_equal(x_transformed, x_transformed_by_loaded_preprocessing)


def test_iris_keras_conda_env(iris_data, keras_model, model_path):
    import cloudpickle
    import tensorflow

    x, y = iris_data

    model = keras_model
    model.fit(x, y)
    cbw.save_model(model_path, model)

    with open(model_path + "/conda.yaml", "r") as f:
        conda_env = yaml.safe_load(f)

    python_version = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    tf_version = tensorflow.__version__
    cloudpickle_version = cloudpickle.__version__

    channels_list = ["defaults", "conda-forge"]
    dependencies = [
        "python={}".format(python_version),
        "pip",
        {
            "pip": [
                "mlflow",
                "cloudpickle=={}".format(cloudpickle_version),
                "tensorflow=={}".format(tf_version),
            ]
        },
    ]
    assert conda_env["channels"] == channels_list
    assert conda_env["dependencies"] == dependencies


def test_iris_keras_conda_env_additional_deps(iris_data, keras_model, model_path):
    import cloudpickle
    import tensorflow

    x, y = iris_data
    model = keras_model
    model.fit(x, y)

    conda_channels = ["special_channel", "custom_channel"]
    conda_deps = ["torch=1.6.0", "fake_package=2.1.0"]
    pip_deps = ["fastapi==0.52.1", "my_package==1.23.1"]

    cbw.save_model(
        model_path,
        model,
        additional_conda_channels=conda_channels,
        additional_conda_deps=conda_deps,
        additional_pip_deps=pip_deps,
    )

    with open(model_path + "/conda.yaml", "r") as f:
        conda_env = yaml.safe_load(f)

    python_version = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    tf_version = tensorflow.__version__
    cloudpickle_version = cloudpickle.__version__

    channels_list = ["defaults", "conda-forge", "special_channel", "custom_channel"]
    dependencies = [
        "python={}".format(python_version),
        "torch=1.6.0",
        "fake_package=2.1.0",
        "pip",
        {
            "pip": [
                "mlflow",
                "cloudpickle=={}".format(cloudpickle_version),
                "fastapi==0.52.1",
                "my_package==1.23.1",
                "tensorflow=={}".format(tf_version),
            ]
        },
    ]
    assert conda_env["channels"] == channels_list
    assert conda_env["dependencies"] == dependencies


def test_iris_keras_conda_env_additional_channels_with_duplicates(
    iris_data, keras_model, model_path
):
    x, y = iris_data
    model = keras_model
    model.fit(x, y)

    conda_channels = ["special_channel", "custom_channel", "custom_channel"]
    with pytest.raises(ValueError):
        cbw.save_model(
            model_path,
            model,
            additional_conda_channels=conda_channels,
        )


def test_iris_keras_conda_env_additional_conda_deps_with_duplicates(
    iris_data, keras_model, model_path
):
    x, y = iris_data
    model = keras_model
    model.fit(x, y)

    conda_deps = ["torch=1.6.0", "torch=1.6.2"]
    with pytest.raises(ValueError):
        cbw.save_model(model_path, model, additional_conda_deps=conda_deps)


def test_iris_keras_conda_env_additional_pip_deps_with_duplicates(
    iris_data, keras_model, model_path
):
    x, y = iris_data
    model = keras_model
    model.fit(x, y)

    pip_deps = ["torch==1.6.0", "torch==1.6.2"]
    with pytest.raises(ValueError):
        cbw.save_model(model_path, model, additional_pip_deps=pip_deps)


def test_iris_keras_conda_env_additional_conda_and_pip_deps_with_common_deps(
    iris_data, keras_model, model_path
):
    x, y = iris_data
    model = keras_model
    model.fit(x, y)

    conda_deps = ["torch=1.6.0", "tensorflow=2.1.0"]
    pip_deps = ["torch==1.6.3", "fastapi>=0.52.1"]

    with pytest.raises(ValueError):
        cbw.save_model(
            model_path,
            model,
            additional_conda_deps=conda_deps,
            additional_pip_deps=pip_deps,
        )"""