import os
from sys import version_info

import pytest
import yaml

import numpy as np

import sklearn.datasets as datasets
import sklearn.preprocessing as sk_preprocessing

import xgboost as xgb

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


def test_iris_xgboost_no_preprocessing(iris_data, model_path):
    x, y = iris_data
    model = xgb.XGBClassifier()
    fitted_model = model.fit(x, y)
    cbw.save_model(model_path, fitted_model)

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict_proba(x)
    loaded_model_predictions = loaded_model.predict(x)

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
def test_iris_xgboost_preprocessing(sk_transformer, iris_data, model_path):
    x, y = iris_data
    x_transformed = sk_transformer.fit_transform(x)

    model = xgb.XGBClassifier()
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(model_path, fitted_model, sk_transformer)

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict_proba(x_transformed)
    loaded_model_predictions = loaded_model.predict(x)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_xgboost_preprocessing_with_function_transformer(
    sk_function_transformer, iris_data, model_path
):
    x, y = iris_data
    x_transformed = sk_function_transformer.fit_transform(x)

    model = xgb.XGBClassifier()
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(model_path, fitted_model, sk_function_transformer)

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict_proba(x_transformed)
    loaded_model_predictions = loaded_model.predict(x)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_xgboost_preprocessing_with_custom_transformer(
    custom_transformer, iris_data, model_path
):
    x, y = iris_data
    x_transformed = custom_transformer(x)

    model = xgb.XGBClassifier()
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(model_path, fitted_model, custom_transformer)

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict_proba(x_transformed)
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
def test_iris_xgboost_data_cleaning_and_preprocessing(
    preprocessor, iris_data, drop_column_transformer, model_path
):
    x, y = iris_data
    x_cleaned = drop_column_transformer(x)
    x_transformed = preprocessor.fit_transform(x_cleaned)

    model = xgb.XGBClassifier()
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(model_path, fitted_model, preprocessor, drop_column_transformer)

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict_proba(x_transformed)
    loaded_model_predictions = loaded_model.predict(x)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_xgboost_data_cleaning_without_preprocessing(iris_data, model_path):
    x, y = iris_data
    sk_transformer = sk_preprocessing.StandardScaler()
    x_transformed = sk_transformer.fit_transform(x)

    model = xgb.XGBClassifier()
    fitted_model = model.fit(x_transformed, y)

    with pytest.raises(ValueError):
        cbw.save_model(model_path, fitted_model, data_cleaning=drop_column_transformer)


def test_iris_xgboost_load_preprocessing_without_preprocessing(iris_data, model_path):
    x, y = iris_data
    model = xgb.XGBClassifier()
    fitted_model = model.fit(x, y)
    cbw.save_model(model_path, fitted_model)

    with pytest.raises(FileNotFoundError):
        loaded_model, preprocessing = cbw.load_model_preprocessing(model_path)


def test_iris_xgboost_load_data_cleaning_without_data_cleaning(iris_data, model_path):
    x, y = iris_data
    sk_transformer = sk_preprocessing.StandardScaler()
    x_transformed = sk_transformer.fit_transform(x)

    model = xgb.XGBClassifier()
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(model_path, fitted_model, sk_transformer)

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
def test_iris_xgboost_get_preprocessed_data(preprocessor, iris_data, model_path):
    x, y = iris_data
    x_transformed = preprocessor.fit_transform(x)

    model = xgb.XGBClassifier()
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(model_path, fitted_model, preprocessor)

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
def test_iris_xgboost_get_cleaned_data(
    preprocessor, iris_data, drop_column_transformer, model_path
):
    x, y = iris_data
    x_cleaned = drop_column_transformer(x)
    x_transformed = preprocessor.fit_transform(x_cleaned)

    model = xgb.XGBClassifier()
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(model_path, fitted_model, preprocessor, drop_column_transformer)

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
def test_iris_xgboost_get_cleaned_and_processed_data(
    preprocessor, iris_data, drop_column_transformer, model_path
):
    x, y = iris_data
    x_cleaned = drop_column_transformer(x)
    x_transformed = preprocessor.fit_transform(x_cleaned)

    model = xgb.XGBClassifier()
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(model_path, fitted_model, preprocessor, drop_column_transformer)

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


def test_iris_xgboost_conda_env(iris_data, model_path):
    import cloudpickle

    x, y = iris_data
    model = xgb.XGBClassifier()
    fitted_model = model.fit(x, y)
    cbw.save_model(model_path, fitted_model)

    with open(model_path + "/conda.yaml", "r") as f:
        conda_env = yaml.safe_load(f)

    python_version = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    xgb_version = xgb.__version__
    cloudpickle_version = cloudpickle.__version__

    channels_list = ["defaults", "conda-forge"]
    dependencies = [
        "python={}".format(python_version),
        "xgboost={}".format(xgb_version),
        "pip",
        {"pip": ["mlflow", "cloudpickle=={}".format(cloudpickle_version)]},
    ]
    assert conda_env["channels"] == channels_list
    assert conda_env["dependencies"] == dependencies


def test_iris_xgboost_conda_env_additional_deps(iris_data, model_path):
    import cloudpickle

    x, y = iris_data
    model = xgb.XGBClassifier()
    fitted_model = model.fit(x, y)

    conda_channels = ["special_channel", "custom_channel"]
    conda_deps = ["torch=1.6.0", "tensorflow=2.1.0"]
    pip_deps = ["fastapi==0.52.1", "my_package==1.23.1"]

    cbw.save_model(
        model_path,
        fitted_model,
        additional_conda_channels=conda_channels,
        additional_conda_deps=conda_deps,
        additional_pip_deps=pip_deps,
    )

    with open(model_path + "/conda.yaml", "r") as f:
        conda_env = yaml.safe_load(f)

    python_version = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    xgb_version = xgb.__version__
    cloudpickle_version = cloudpickle.__version__

    channels_list = ["defaults", "conda-forge", "special_channel", "custom_channel"]
    dependencies = [
        "python={}".format(python_version),
        "torch=1.6.0",
        "tensorflow=2.1.0",
        "xgboost={}".format(xgb_version),
        "pip",
        {
            "pip": [
                "mlflow",
                "cloudpickle=={}".format(cloudpickle_version),
                "fastapi==0.52.1",
                "my_package==1.23.1",
            ]
        },
    ]
    assert conda_env["channels"] == channels_list
    assert conda_env["dependencies"] == dependencies


def test_iris_xgboost_conda_env_additional_channels_with_duplicates(
    iris_data, model_path
):
    x, y = iris_data
    model = xgb.XGBClassifier()
    fitted_model = model.fit(x, y)

    conda_channels = ["special_channel", "custom_channel", "custom_channel"]
    with pytest.raises(ValueError):
        cbw.save_model(
            model_path,
            fitted_model,
            additional_conda_channels=conda_channels,
        )


def test_iris_xgboost_conda_env_additional_conda_deps_with_duplicates(
    iris_data, model_path
):
    x, y = iris_data
    model = xgb.XGBClassifier()
    fitted_model = model.fit(x, y)

    conda_deps = ["torch=1.6.0", "torch=1.6.2"]
    with pytest.raises(ValueError):
        cbw.save_model(model_path, fitted_model, additional_conda_deps=conda_deps)


def test_iris_xgboost_conda_env_additional_pip_deps_with_duplicates(
    iris_data, model_path
):
    x, y = iris_data
    model = xgb.XGBClassifier()
    fitted_model = model.fit(x, y)

    pip_deps = ["torch==1.6.0", "torch==1.6.2"]
    with pytest.raises(ValueError):
        cbw.save_model(model_path, fitted_model, additional_pip_deps=pip_deps)


def test_iris_xgboost_conda_env_additional_conda_and_pip_deps_with_common_deps(
    iris_data, model_path
):
    x, y = iris_data
    model = xgb.XGBClassifier()
    fitted_model = model.fit(x, y)

    conda_deps = ["torch=1.6.0", "tensorflow=2.1.0"]
    pip_deps = ["torch==1.6.3", "fastapi>=0.52.1"]
    with pytest.raises(ValueError):
        cbw.save_model(
            model_path,
            fitted_model,
            additional_conda_deps=conda_deps,
            additional_pip_deps=pip_deps,
        )
