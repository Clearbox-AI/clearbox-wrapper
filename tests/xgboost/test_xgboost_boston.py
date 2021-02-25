import os

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as sk_preprocessing
import xgboost as xgb

import clearbox_wrapper as cbw


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


@pytest.fixture()
def drop_column_transformer():
    def drop_column(dataframe_x):
        x_transformed = dataframe_x.drop("INDUS", axis=1)
        return x_transformed

    return drop_column


def _check_schema(pdf, input_schema):
    if isinstance(pdf, (list, np.ndarray, dict)):
        try:
            pdf = pd.DataFrame(pdf)
        except Exception as e:
            message = (
                "This model contains a model signature, which suggests a DataFrame input."
                "There was an error casting the input data to a DataFrame: {0}".format(
                    str(e)
                )
            )
            raise cbw.ClearboxWrapperException(message)
    if not isinstance(pdf, pd.DataFrame):
        message = (
            "Expected input to be DataFrame or list. Found: %s" % type(pdf).__name__
        )
        raise cbw.ClearboxWrapperException(message)

    if input_schema.has_column_names():
        # make sure there are no missing columns
        col_names = input_schema.column_names()
        expected_names = set(col_names)
        actual_names = set(pdf.columns)
        missing_cols = expected_names - actual_names
        extra_cols = actual_names - expected_names
        # Preserve order from the original columns, since missing/extra columns are likely to
        # be in same order.
        missing_cols = [c for c in col_names if c in missing_cols]
        extra_cols = [c for c in pdf.columns if c in extra_cols]
        if missing_cols:
            print(
                "Model input is missing columns {0}."
                " Note that there were extra columns: {1}".format(
                    missing_cols, extra_cols
                )
            )
            return False
    else:
        if len(pdf.columns) != len(input_schema.columns):
            print(
                "The model signature declares "
                "{0} input columns but the provided input has "
                "{1} columns. Note: the columns were not named in the signature so we can "
                "only verify their count.".format(
                    len(input_schema.columns), len(pdf.columns)
                )
            )
            return False
        col_names = pdf.columns[: len(input_schema.columns)]
    return True


def test_boston_xgboost_no_preprocessing(boston_training_test, model_path):
    x_train, x_test, y_train, _ = boston_training_test
    model = xgb.XGBRegressor()
    fitted_model = model.fit(x_train, y_train)
    cbw.save_model(model_path, fitted_model, zip=False)

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
    cbw.save_model(model_path, fitted_model, preprocessing=sk_transformer, zip=False)

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
    cbw.save_model(
        model_path, fitted_model, preprocessing=sk_function_transformer, zip=False
    )

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
    cbw.save_model(
        model_path, fitted_model, preprocessing=custom_transformer, zip=False
    )

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
def test_boston_xgboost_data_preparation_and_preprocessing(
    preprocessor, add_value_to_column_transformer, boston_training_test, model_path
):
    x_train, x_test, y_train, _ = boston_training_test
    x_train_prepared = add_value_to_column_transformer(x_train)
    x_train_transformed = preprocessor.fit_transform(x_train_prepared)

    model = xgb.XGBRegressor()
    fitted_model = model.fit(x_train_transformed, y_train)
    cbw.save_model(
        model_path,
        fitted_model,
        preprocessing=preprocessor,
        data_preparation=add_value_to_column_transformer,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)
    x_test_prepared = add_value_to_column_transformer(x_test)
    x_test_transformed = preprocessor.transform(x_test_prepared)

    original_model_predictions = fitted_model.predict(x_test_transformed)
    loaded_model_predictions = loaded_model.predict(x_test)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def tests_boston_xgb_zipped_path_already_exists(boston_training_test, model_path):
    x_train, x_test, y_train, _ = boston_training_test
    model = xgb.XGBRegressor()
    fitted_model = model.fit(x_train, y_train)
    cbw.save_model(model_path, fitted_model)
    with pytest.raises(cbw.ClearboxWrapperException):
        cbw.save_model(model_path, fitted_model)


def tests_boston_xgb_path_already_exists(boston_training_test, model_path):
    x_train, x_test, y_train, _ = boston_training_test
    model = xgb.XGBRegressor()
    fitted_model = model.fit(x_train, y_train)
    cbw.save_model(model_path, fitted_model, zip=False)
    with pytest.raises(cbw.ClearboxWrapperException):
        cbw.save_model(model_path, fitted_model, zip=False)


def test_boston_xgb_no_preprocessing_check_model_signature(
    boston_training_test, model_path
):
    x_train, x_test, y_train, _ = boston_training_test
    model = xgb.XGBRegressor()
    fitted_model = model.fit(x_train, y_train)
    cbw.save_model(model_path, fitted_model, input_data=x_train, zip=False)
    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict(x_train[:5])
    loaded_model_predictions = loaded_model.predict(x_train[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)

    mlmodel = cbw.Model.load(model_path)
    assert _check_schema(x_train, mlmodel.get_model_input_schema())


@pytest.mark.parametrize(
    "preprocessor",
    [
        (sk_preprocessing.StandardScaler()),
        (sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50)),
        (sk_preprocessing.KBinsDiscretizer(n_bins=2, encode="ordinal")),
    ],
)
def test_boston_xgb_preprocessing_check_model_and_preprocessing_signature(
    preprocessor, boston_training_test, model_path
):
    x_train, x_test, y_train, _ = boston_training_test
    x_train_transformed = preprocessor.fit_transform(x_train)

    model = xgb.XGBRegressor()
    fitted_model = model.fit(x_train_transformed, y_train)
    cbw.save_model(
        model_path,
        fitted_model,
        preprocessing=preprocessor,
        input_data=x_train,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = fitted_model.predict(x_train_transformed[:5])
    loaded_model_predictions = loaded_model.predict(x_train[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)

    mlmodel = cbw.Model.load(model_path)
    preprocessing_input_schema = mlmodel.get_preprocessing_input_schema()
    preprocessing_output_schema = mlmodel.get_preprocessing_output_schema()
    model_input_schema = mlmodel.get_model_input_schema()

    assert _check_schema(x_train, preprocessing_input_schema)
    assert _check_schema(x_train_transformed, preprocessing_output_schema)
    assert _check_schema(x_train_transformed, model_input_schema)
    assert preprocessing_output_schema == model_input_schema


@pytest.mark.parametrize(
    "preprocessor",
    [
        (sk_preprocessing.StandardScaler()),
        (sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50)),
        (sk_preprocessing.KBinsDiscretizer(n_bins=2, encode="ordinal")),
    ],
)
def test_boston_xgb_check_model_preprocessing_and_data_preparation_signature(
    preprocessor, boston_training_test, drop_column_transformer, model_path
):
    x_train, x_test, y_train, _ = boston_training_test
    print(x_train.columns)
    x_train_prepared = drop_column_transformer(x_train)
    x_train_transformed = preprocessor.fit_transform(x_train_prepared)

    model = xgb.XGBRegressor()
    fitted_model = model.fit(x_train_transformed, y_train)
    cbw.save_model(
        model_path,
        fitted_model,
        preprocessing=preprocessor,
        data_preparation=drop_column_transformer,
        input_data=x_train,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)
    x_test_prepared = drop_column_transformer(x_test)
    x_test_transformed = preprocessor.transform(x_test_prepared)

    original_model_predictions = fitted_model.predict(x_test_transformed)
    loaded_model_predictions = loaded_model.predict(x_test)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)

    mlmodel = cbw.Model.load(model_path)
    data_preparation_input_schema = mlmodel.get_data_preparation_input_schema()
    data_preparation_output_schema = mlmodel.get_data_preparation_output_schema()
    preprocessing_input_schema = mlmodel.get_preprocessing_input_schema()
    preprocessing_output_schema = mlmodel.get_preprocessing_output_schema()
    model_input_schema = mlmodel.get_model_input_schema()

    assert _check_schema(x_train, data_preparation_input_schema)
    assert _check_schema(x_train_prepared, data_preparation_output_schema)
    assert _check_schema(x_train_prepared, preprocessing_input_schema)
    assert _check_schema(x_train_transformed, preprocessing_output_schema)
    assert _check_schema(x_train_transformed, model_input_schema)
    assert not _check_schema(x_train, model_input_schema)
    assert data_preparation_output_schema == preprocessing_input_schema
    assert preprocessing_output_schema == model_input_schema
