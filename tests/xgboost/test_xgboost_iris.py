import os
from sys import version_info

import numpy as np
import pytest
import sklearn.datasets as datasets
import sklearn.preprocessing as sk_preprocessing
import xgboost as xgb
import yaml

import clearbox_wrapper as cbw


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
    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x, y)
    cbw.save_model(model_path, fitted_model, zip=False)

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict_proba(x)
    loaded_model_predictions = loaded_model.predict_proba(x)

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

    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(model_path, fitted_model, preprocessing=sk_transformer, zip=False)

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict(x_transformed)
    loaded_model_predictions = loaded_model.predict(x)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_xgboost_preprocessing_with_function_transformer(
    sk_function_transformer, iris_data, model_path
):
    x, y = iris_data
    x_transformed = sk_function_transformer.fit_transform(x)

    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(
        model_path, fitted_model, preprocessing=sk_function_transformer, zip=False
    )

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict_proba(x_transformed)
    loaded_model_predictions = loaded_model.predict_proba(x)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_xgboost_preprocessing_with_custom_transformer(
    custom_transformer, iris_data, model_path
):
    x, y = iris_data
    x_transformed = custom_transformer(x)

    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(
        model_path, fitted_model, preprocessing=custom_transformer, zip=False
    )

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict(x_transformed)
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
def test_iris_xgboost_data_preparation_and_preprocessing(
    preprocessor, iris_data, drop_column_transformer, model_path
):
    x, y = iris_data
    x_prepared = drop_column_transformer(x)
    x_transformed = preprocessor.fit_transform(x_prepared)

    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(
        model_path,
        fitted_model,
        preprocessing=preprocessor,
        data_preparation=drop_column_transformer,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict_proba(x_transformed)
    loaded_model_predictions = loaded_model.predict_proba(x)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_xgboost_data_preparation_without_preprocessing(iris_data, model_path):
    x, y = iris_data
    sk_transformer = sk_preprocessing.StandardScaler()
    x_transformed = sk_transformer.fit_transform(x)

    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x_transformed, y)

    with pytest.raises(ValueError):
        cbw.save_model(
            model_path,
            fitted_model,
            data_preparation=drop_column_transformer,
            zip=False,
        )


def test_iris_xgboost_load_preprocessing_without_preprocessing(iris_data, model_path):
    x, y = iris_data
    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x, y)
    cbw.save_model(model_path, fitted_model, zip=False)
    loaded_model = cbw.load_model(model_path)
    with pytest.raises(cbw.ClearboxWrapperException):
        loaded_model.preprocess_data(x)


def test_iris_xgboost_load_data_preparation_without_data_preparation(
    iris_data, model_path
):
    x, y = iris_data
    sk_transformer = sk_preprocessing.StandardScaler()
    x_transformed = sk_transformer.fit_transform(x)

    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(model_path, fitted_model, preprocessing=sk_transformer, zip=False)
    loaded_model = cbw.load_model(model_path)
    with pytest.raises(cbw.ClearboxWrapperException):
        loaded_model.prepare_data(x)


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

    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(model_path, fitted_model, preprocessing=preprocessor, zip=False)
    loaded_model = cbw.load_model(model_path)
    x_transformed_by_loaded_preprocessing = loaded_model.preprocess_data(x)
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
def test_iris_xgboost_get_prepared_data(
    preprocessor, iris_data, drop_column_transformer, model_path
):
    x, y = iris_data
    x_prepared = drop_column_transformer(x)
    x_transformed = preprocessor.fit_transform(x_prepared)

    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(
        model_path,
        fitted_model,
        preprocessing=preprocessor,
        data_preparation=drop_column_transformer,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)
    x_prepared_by_loaded_data_preparation = loaded_model.prepare_data(x)
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
def test_iris_xgboost_get_prepared_and_processed_data(
    preprocessor, iris_data, drop_column_transformer, model_path
):
    x, y = iris_data
    x_prepared = drop_column_transformer(x)
    x_transformed = preprocessor.fit_transform(x_prepared)

    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(
        model_path,
        fitted_model,
        preprocessing=preprocessor,
        data_preparation=drop_column_transformer,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)
    x_prepared_by_loaded_data_preparation = loaded_model.prepare_data(x)
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
def test_iris_xgboost_predict_without_preprocessing(
    preprocessor, iris_data, drop_column_transformer, model_path
):
    x, y = iris_data
    x_prepared = drop_column_transformer(x)
    x_transformed = preprocessor.fit_transform(x_prepared)

    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(
        model_path,
        fitted_model,
        preprocessing=preprocessor,
        data_preparation=drop_column_transformer,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict_proba(x_transformed)
    loaded_model_predictions = loaded_model.predict_proba(x, preprocess=False)
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
def test_iris_xgboost_predict_without_data_preparation(
    preprocessor, iris_data, drop_column_transformer, model_path
):
    x, y = iris_data
    x_prepared = drop_column_transformer(x)
    x_transformed = preprocessor.fit_transform(x_prepared)

    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x_transformed, y)
    cbw.save_model(
        model_path,
        fitted_model,
        preprocessing=preprocessor,
        data_preparation=drop_column_transformer,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)
    with pytest.raises(ValueError):
        loaded_model.predict(x, prepare_data=False)


def test_iris_xgboost_conda_env(iris_data, model_path):
    import dill

    x, y = iris_data
    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x, y)
    cbw.save_model(model_path, fitted_model, zip=False)

    with open(model_path + "/conda.yaml", "r") as f:
        conda_env = yaml.safe_load(f)

    python_version = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    xgb_version = xgb.__version__
    dill_version = dill.__version__

    channels_list = ["defaults", "conda-forge"]
    dependencies = [
        "python={}".format(python_version),
        "pip",
        {
            "pip": [
                "dill=={}".format(dill_version),
                "xgboost=={}".format(xgb_version),
            ]
        },
    ]
    assert conda_env["channels"] == channels_list
    assert conda_env["dependencies"] == dependencies


def test_iris_xgboost_conda_env_additional_deps(iris_data, model_path):
    import dill

    x, y = iris_data
    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x, y)

    add_deps = [
        "torch==1.6.0",
        "tensorflow==2.1.0",
        "fastapi==0.52.1",
        "my_package==1.23.1",
    ]

    cbw.save_model(model_path, fitted_model, additional_deps=add_deps, zip=False)

    with open(model_path + "/conda.yaml", "r") as f:
        conda_env = yaml.safe_load(f)

    python_version = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    xgb_version = xgb.__version__
    dill_version = dill.__version__

    channels_list = ["defaults", "conda-forge"]
    dependencies = [
        "python={}".format(python_version),
        "pip",
        {
            "pip": [
                "dill=={}".format(dill_version),
                "torch==1.6.0",
                "tensorflow==2.1.0",
                "fastapi==0.52.1",
                "my_package==1.23.1",
                "xgboost=={}".format(xgb_version),
            ]
        },
    ]
    assert conda_env["channels"] == channels_list
    assert conda_env["dependencies"] == dependencies


def test_iris_xgboost_conda_env_additional_deps_with_duplicates(iris_data, model_path):
    x, y = iris_data
    model = xgb.XGBClassifier(use_label_encoder=False)
    fitted_model = model.fit(x, y)

    add_deps = ["torch==1.6.0", "torch==1.6.2"]
    with pytest.raises(ValueError):
        cbw.save_model(model_path, fitted_model, additional_deps=add_deps, zip=False)
