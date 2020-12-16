import os

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as sk_preprocessing
import xgboost as xgb

import clearbox_wrapper.clearbox_wrapper as cbw


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture(scope="module")
def boston_training_test():
    csv_path = "tests/datasets/boston_housing.csv"
    target_column = "MEDV"
    boston_dataset = pd.read_csv(csv_path)
    y = boston_dataset[target_column]
    x = boston_dataset.drop(target_column, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test


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


def test_boston_xgboost_no_preprocessing(boston_training_test, model_path):
    x_train, x_test, y_train, _ = boston_training_test
    model = xgb.XGBRegressor()
    fitted_model = model.fit(x_train, y_train)
    cbw.save_model(model_path, fitted_model)

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict(x_test)
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
def test_boston_xgboost_preprocessing(sk_transformer, boston_training_test, model_path):
    x_train, x_test, y_train, _ = boston_training_test
    x_train_transformed = sk_transformer.fit_transform(x_train)

    model = xgb.XGBRegressor()
    fitted_model = model.fit(x_train_transformed, y_train)
    cbw.save_model(model_path, fitted_model, sk_transformer)

    loaded_model = cbw.load_model(model_path)
    x_test_transformed = sk_transformer.transform(x_test)
    original_model_predictions = fitted_model.predict(x_test_transformed)
    loaded_model_predictions = loaded_model.predict(x_test)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_boston_xgboost_preprocessing_with_function_transformer(
    sk_function_transformer, boston_training_test, model_path
):
    x_train, x_test, y_train, _ = boston_training_test
    x_train_transformed = sk_function_transformer.fit_transform(x_train)

    model = xgb.XGBRegressor()
    fitted_model = model.fit(x_train_transformed, y_train)
    cbw.save_model(model_path, fitted_model, sk_function_transformer)

    loaded_model = cbw.load_model(model_path)
    x_test_transformed = sk_function_transformer.transform(x_test)
    original_model_predictions = fitted_model.predict(x_test_transformed)
    loaded_model_predictions = loaded_model.predict(x_test)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_boston_xgboost_preprocessing_with_custom_transformer(
    custom_transformer, boston_training_test, model_path
):
    x_train, x_test, y_train, _ = boston_training_test
    x_train_transformed = custom_transformer(x_train)

    model = xgb.XGBRegressor()
    fitted_model = model.fit(x_train_transformed, y_train)
    cbw.save_model(model_path, fitted_model, custom_transformer)

    loaded_model = cbw.load_model(model_path)
    x_test_transformed = custom_transformer(x_test)
    original_model_predictions = fitted_model.predict(x_test_transformed)
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
def test_boston_xgboost_data_cleaning_and_preprocessing(
    preprocessor, add_value_to_column_transformer, boston_training_test, model_path
):
    x_train, x_test, y_train, _ = boston_training_test
    x_train_cleaned = add_value_to_column_transformer(x_train)
    x_train_transformed = preprocessor.fit_transform(x_train_cleaned)

    model = xgb.XGBRegressor()
    fitted_model = model.fit(x_train_transformed, y_train)
    cbw.save_model(
        model_path, fitted_model, preprocessor, add_value_to_column_transformer
    )

    loaded_model = cbw.load_model(model_path)
    x_test_cleaned = add_value_to_column_transformer(x_test)
    x_test_transformed = preprocessor.transform(x_test_cleaned)

    original_model_predictions = fitted_model.predict(x_test_transformed)
    loaded_model_predictions = loaded_model.predict(x_test)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)
