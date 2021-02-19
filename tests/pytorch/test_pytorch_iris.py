import os
from sys import version_info

import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing as sk_preprocessing
import torch
import torch.nn as nn
import yaml

import clearbox_wrapper as cbw


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture()
def sk_function_transformer():
    def simple_preprocessor(data_x):
        return data_x ** 2

    transformer = sk_preprocessing.FunctionTransformer(
        simple_preprocessor, validate=True
    )
    return transformer


@pytest.fixture()
def custom_transformer():
    def simple_preprocessor(data_x):
        transformed_x = data_x + 1.0
        return transformed_x

    return simple_preprocessor


@pytest.fixture()
def add_value_to_column_transformer():
    def drop_column(dataframe_x):
        x_transformed = dataframe_x + dataframe_x
        return x_transformed

    return drop_column


@pytest.fixture(scope="module")
def iris_training():
    csv_path = "tests/datasets/iris_training_set_one_hot_y.csv"
    iris_dataset = pd.read_csv(csv_path)
    x = iris_dataset.iloc[:, :4]
    y = iris_dataset.iloc[:, 4:]
    return x, y


@pytest.fixture(scope="module")
def iris_test():
    csv_path = "tests/datasets/iris_test_set_one_hot_y.csv"
    iris_dataset = pd.read_csv(csv_path)
    x = iris_dataset.iloc[:, :4]
    y = iris_dataset.iloc[:, 4:]
    return x, y


@pytest.fixture()
def iris_pytorch_model():
    D_in, H, D_out = 4, 8, 3
    model = nn.Sequential(
        nn.Linear(D_in, H),
        nn.ReLU(),
        nn.Linear(H, D_out),
        nn.Softmax(dim=0),
    )
    device = torch.device("cpu")
    model.to(device)
    return model


def iris_pytorch_model_training(model, x_train, y_train):

    learning_rate = 1e-4
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    idx = np.arange(x_train.size()[0])
    for _epoch in range(10):
        np.random.shuffle(idx)
        for id in idx:
            y_pred = model(x_train[id])
            y = y_train[id]
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test_iris_pytorch_no_preprocessing(
    iris_training, iris_test, iris_pytorch_model, model_path
):
    x_train, y_train = iris_training
    x_test, _ = iris_test

    x_train = torch.Tensor(x_train.values)
    y_train = torch.Tensor(y_train.values)

    x_test_tensor = torch.Tensor(x_test.values)

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_train, y_train)

    cbw.save_model(model_path, model, zip=False)
    loaded_model = cbw.load_model(model_path)

    original_model_predictions = model(x_test_tensor).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test)

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
def test_iris_pytorch_preprocessing(
    sk_transformer, iris_training, iris_test, iris_pytorch_model, model_path
):

    x_train, y_train = iris_training
    x_test, _ = iris_test

    x_transformed = sk_transformer.fit_transform(x_train)
    x_transformed = torch.Tensor(x_transformed)
    y_train_tensor = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = sk_transformer.transform(x_data)
        return x_transformed

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_transformed, y_train_tensor)

    cbw.save_model(model_path, model, preprocessing=preprocessing_function, zip=False)
    loaded_model = cbw.load_model(model_path)

    x_test_transformed = preprocessing_function(x_test)
    x_test_transformed = torch.Tensor(x_test_transformed)
    original_model_predictions = model(x_test_transformed).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test)

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_pytorch_preprocessing_with_function_transformer(
    sk_function_transformer, iris_training, iris_test, iris_pytorch_model, model_path
):
    x_train, y_train = iris_training
    x_test, _ = iris_test

    x_transformed = sk_function_transformer.fit_transform(x_train)
    x_transformed = torch.Tensor(x_transformed)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = sk_function_transformer.transform(x_data)
        return x_transformed

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_transformed, y_train)

    cbw.save_model(model_path, model, preprocessing=preprocessing_function, zip=False)
    loaded_model = cbw.load_model(model_path)

    x_test_transformed = preprocessing_function(x_test)
    x_test_transformed = torch.Tensor(x_test_transformed)
    original_model_predictions = model(x_test_transformed).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test)

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_pytorch_preprocessing_with_custom_transformer(
    custom_transformer, iris_training, iris_test, iris_pytorch_model, model_path
):
    x_train, y_train = iris_training
    x_test, _ = iris_test

    x_transformed = custom_transformer(x_train)
    x_transformed = torch.Tensor(x_transformed.values)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = custom_transformer(x_data)
        return x_transformed

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_transformed, y_train)

    cbw.save_model(model_path, model, preprocessing=preprocessing_function, zip=False)
    loaded_model = cbw.load_model(model_path)

    x_test_transformed = preprocessing_function(x_test)
    x_test_transformed = torch.Tensor(x_test_transformed.values)
    original_model_predictions = model(x_test_transformed).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test)

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
def test_iris_pytorch_data_preparation_and_preprocessing(
    preprocessor,
    add_value_to_column_transformer,
    iris_training,
    iris_test,
    iris_pytorch_model,
    model_path,
):
    x_train, y_train = iris_training
    x_test, _ = iris_test

    x_prepared = add_value_to_column_transformer(x_train)
    x_transformed = preprocessor.fit_transform(x_prepared)

    x_transformed = torch.Tensor(x_transformed)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = preprocessor.transform(x_data)
        return x_transformed

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_transformed, y_train)

    cbw.save_model(
        model_path,
        model,
        preprocessing=preprocessing_function,
        data_preparation=add_value_to_column_transformer,
        zip=False,
    )
    loaded_model = cbw.load_model(model_path)

    x_test_prepared = add_value_to_column_transformer(x_test)
    x_test_transformed = preprocessing_function(x_test_prepared)
    x_test_transformed = torch.Tensor(x_test_transformed)

    original_model_predictions = model(x_test_transformed).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test)

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_pytorch_data_preparation_without_preprocessing(
    add_value_to_column_transformer,
    iris_training,
    iris_test,
    iris_pytorch_model,
    model_path,
):
    x_train, y_train = iris_training

    sk_transformer = sk_preprocessing.StandardScaler()
    x_transformed = sk_transformer.fit_transform(x_train)

    x_transformed = torch.Tensor(x_transformed)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = sk_transformer.transform(x_data)
        return x_transformed

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_transformed, y_train)

    with pytest.raises(ValueError):
        cbw.save_model(
            model_path,
            model,
            data_preparation=add_value_to_column_transformer,
            zip=False,
        )


def test_iris_pytorch_load_preprocessing_without_preprocessing(
    iris_training, iris_test, iris_pytorch_model, model_path
):
    x_train, y_train = iris_training

    x_train_tensor = torch.Tensor(x_train.values)
    y_train = torch.Tensor(y_train.values)

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_train_tensor, y_train)

    cbw.save_model(model_path, model, zip=False)
    loaded_model = cbw.load_model(model_path)
    with pytest.raises(cbw.ClearboxWrapperException):
        loaded_model.preprocess_data(x_train)


def test_iris_pytorch_load_data_preparation_without_data_preparation(
    iris_training, iris_test, iris_pytorch_model, model_path
):
    x_train, y_train = iris_training

    sk_transformer = sk_preprocessing.StandardScaler()
    x_transformed = sk_transformer.fit_transform(x_train)

    x_transformed = torch.Tensor(x_transformed)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = sk_transformer.transform(x_data)
        return torch.Tensor(x_transformed)

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_transformed, y_train)

    cbw.save_model(model_path, model, preprocessing=preprocessing_function, zip=False)
    loaded_model = cbw.load_model(model_path)
    with pytest.raises(cbw.ClearboxWrapperException):
        loaded_model.prepare_data(x_train)


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
def test_iris_pytorch_get_preprocessed_data(
    preprocessor, iris_training, iris_test, iris_pytorch_model, model_path
):
    x_train, y_train = iris_training

    x_transformed = preprocessor.fit_transform(x_train)
    x_transformed_tensor = torch.Tensor(x_transformed)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = preprocessor.transform(x_data)
        return x_transformed

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_transformed_tensor, y_train)

    cbw.save_model(model_path, model, preprocessing=preprocessing_function, zip=False)
    loaded_model = cbw.load_model(model_path)
    x_transformed_by_loaded_preprocessing = loaded_model.preprocess_data(x_train)
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
def test_iris_pytorch_get_prepared_data(
    preprocessor,
    add_value_to_column_transformer,
    iris_training,
    iris_test,
    iris_pytorch_model,
    model_path,
):
    x_train, y_train = iris_training

    x_prepared = add_value_to_column_transformer(x_train)
    x_transformed = preprocessor.fit_transform(x_prepared)

    x_transformed_tensor = torch.Tensor(x_transformed)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = preprocessor.transform(x_data)
        return x_transformed

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_transformed_tensor, y_train)

    cbw.save_model(
        model_path,
        model,
        preprocessing=preprocessing_function,
        data_preparation=add_value_to_column_transformer,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)
    x_prepared_by_loaded_data_preparation = loaded_model.prepare_data(x_train)
    np.testing.assert_array_equal(x_prepared, x_prepared_by_loaded_data_preparation)


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
def test_iris_pytorch_get_prepared_and_processed_data(
    preprocessor,
    add_value_to_column_transformer,
    iris_training,
    iris_test,
    iris_pytorch_model,
    model_path,
):
    x_train, y_train = iris_training

    x_prepared = add_value_to_column_transformer(x_train)
    x_transformed = preprocessor.fit_transform(x_prepared)

    x_transformed_tensor = torch.Tensor(x_transformed)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = preprocessor.transform(x_data)
        return x_transformed

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_transformed_tensor, y_train)

    cbw.save_model(
        model_path,
        model,
        preprocessing=preprocessing_function,
        data_preparation=add_value_to_column_transformer,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)
    x_prepared_by_loaded_data_preparation = loaded_model.prepare_data(x_train)
    x_transformed_by_loaded_preprocessing = loaded_model.preprocess_data(
        x_prepared_by_loaded_data_preparation
    )
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
def test_iris_pytorch_predict_without_preprocessing(
    preprocessor,
    add_value_to_column_transformer,
    iris_training,
    iris_test,
    iris_pytorch_model,
    model_path,
):
    x_train, y_train = iris_training
    x_test, _ = iris_test

    x_prepared = add_value_to_column_transformer(x_train)
    x_transformed = preprocessor.fit_transform(x_prepared)

    x_transformed = torch.Tensor(x_transformed)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = preprocessor.transform(x_data)
        return x_transformed

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_transformed, y_train)

    cbw.save_model(
        model_path,
        model,
        preprocessing=preprocessing_function,
        data_preparation=add_value_to_column_transformer,
        zip=False,
    )
    loaded_model = cbw.load_model(model_path)

    x_test_prepared = add_value_to_column_transformer(x_test)
    x_test_transformed = preprocessing_function(x_test_prepared)
    x_test_transformed = torch.Tensor(x_test_transformed)

    original_model_predictions = model(x_test_transformed).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test, preprocess=False)

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        original_model_predictions,
        loaded_model_predictions,
    )


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
def test_iris_pytorch_predict_without_data_preparation(
    preprocessor,
    add_value_to_column_transformer,
    iris_training,
    iris_test,
    iris_pytorch_model,
    model_path,
):
    x_train, y_train = iris_training
    x_test, _ = iris_test

    x_prepared = add_value_to_column_transformer(x_train)
    x_transformed = preprocessor.fit_transform(x_prepared)

    x_transformed = torch.Tensor(x_transformed)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = preprocessor.transform(x_data)
        return x_transformed

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_transformed, y_train)

    cbw.save_model(
        model_path,
        model,
        preprocessing=preprocessing_function,
        data_preparation=add_value_to_column_transformer,
        zip=False,
    )
    loaded_model = cbw.load_model(model_path)

    x_test_prepared = add_value_to_column_transformer(x_test)
    x_test_transformed = preprocessing_function(x_test_prepared)
    x_test_transformed = torch.Tensor(x_test_transformed)

    original_model_predictions = model(x_test_transformed).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test, prepare_data=False)

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        original_model_predictions,
        loaded_model_predictions,
    )


def test_iris_pytorch_conda_env(
    iris_training, iris_test, iris_pytorch_model, model_path
):
    import cloudpickle

    x_train, y_train = iris_training

    x_train = torch.Tensor(x_train.values)
    y_train = torch.Tensor(y_train.values)

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_train, y_train)

    cbw.save_model(model_path, model, zip=False)

    with open(model_path + "/conda.yaml", "r") as f:
        conda_env = yaml.safe_load(f)

    python_version = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    pytorch_version = torch.__version__
    cloudpickle_version = cloudpickle.__version__

    channels_list = ["defaults", "conda-forge"]
    dependencies = [
        "python={}".format(python_version),
        "pip",
        {
            "pip": [
                "cloudpickle=={}".format(cloudpickle_version),
                "torch=={}".format(pytorch_version),
            ]
        },
    ]
    assert conda_env["channels"] == channels_list
    assert conda_env["dependencies"] == dependencies


def test_iris_pytorch_conda_env_additional_deps(
    iris_training, iris_test, iris_pytorch_model, model_path
):
    import cloudpickle

    x_train, y_train = iris_training

    x_train = torch.Tensor(x_train.values)
    y_train = torch.Tensor(y_train.values)

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_train, y_train)

    add_deps = [
        "keras==1.6.0",
        "fake_package==2.1.0",
        "fastapi==0.52.1",
        "my_package==1.23.1",
    ]

    cbw.save_model(model_path, model, additional_deps=add_deps, zip=False)

    with open(model_path + "/conda.yaml", "r") as f:
        conda_env = yaml.safe_load(f)

    python_version = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    pytorch_version = torch.__version__
    cloudpickle_version = cloudpickle.__version__

    channels_list = ["defaults", "conda-forge"]
    dependencies = [
        "python={}".format(python_version),
        "pip",
        {
            "pip": [
                "cloudpickle=={}".format(cloudpickle_version),
                "keras==1.6.0",
                "fake_package==2.1.0",
                "fastapi==0.52.1",
                "my_package==1.23.1",
                "torch=={}".format(pytorch_version),
            ]
        },
    ]
    assert conda_env["channels"] == channels_list
    assert conda_env["dependencies"] == dependencies


def test_iris_pytorch_conda_env_additional_deps_with_duplicates(
    iris_training, iris_test, iris_pytorch_model, model_path
):
    x_train, y_train = iris_training

    x_train = torch.Tensor(x_train.values)
    y_train = torch.Tensor(y_train.values)

    model = iris_pytorch_model
    iris_pytorch_model_training(model, x_train, y_train)

    add_deps = ["keras==1.6.0", "keras==1.6.2"]
    with pytest.raises(ValueError):
        cbw.save_model(model_path, model, additional_deps=add_deps, zip=False)
