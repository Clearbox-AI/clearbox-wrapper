import os

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as sk_preprocessing
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

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
def keras_model():
    keras_clf = Sequential()
    keras_clf.add(Dense(8, input_dim=13, activation="relu"))
    keras_clf.add(Dense(4, activation="relu"))
    keras_clf.add(Dense(1))

    keras_clf.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return keras_clf


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
    def double_dataframe(dataframe_x):
        x_transformed = dataframe_x + dataframe_x
        return x_transformed

    return double_dataframe


def test_boston_keras_no_preprocessing(boston_training_test, keras_model, model_path):

    x_train, x_test, y_train, _ = boston_training_test

    model = keras_model
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    cbw.save_model(model_path, model, zip=False)

    loaded_model = cbw.load_model(model_path)

    original_model_predictions = model.predict(x_test)
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
def test_boston_keras_preprocessing(
    sk_transformer, boston_training_test, keras_model, model_path
):
    x_train, x_test, y_train, _ = boston_training_test
    x_train_transformed = sk_transformer.fit_transform(x_train)

    model = keras_model
    model.fit(x_train_transformed, y_train, epochs=10, batch_size=32)
    cbw.save_model(model_path, model, preprocessing=sk_transformer, zip=False)

    loaded_model = cbw.load_model(model_path)
    x_test_transformed = sk_transformer.transform(x_test)
    original_model_predictions = model.predict(x_test_transformed)
    loaded_model_predictions = loaded_model.predict(x_test)

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_boston_keras_preprocessing_with_function_transformer(
    sk_function_transformer, boston_training_test, keras_model, model_path
):
    x_train, x_test, y_train, _ = boston_training_test
    x_train_transformed = sk_function_transformer.fit_transform(x_train)

    model = keras_model
    model.fit(x_train_transformed, y_train, epochs=10, batch_size=32)
    cbw.save_model(model_path, model, preprocessing=sk_function_transformer, zip=False)

    loaded_model = cbw.load_model(model_path)
    x_test_transformed = sk_function_transformer.transform(x_test)
    original_model_predictions = model.predict(x_test_transformed)
    loaded_model_predictions = loaded_model.predict(x_test)

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_boston_keras_preprocessing_with_custom_transformer(
    custom_transformer, boston_training_test, keras_model, model_path
):
    x_train, x_test, y_train, _ = boston_training_test
    x_train_transformed = custom_transformer(x_train)

    model = keras_model
    model.fit(x_train_transformed, y_train, epochs=10, batch_size=32)
    cbw.save_model(model_path, model, preprocessing=custom_transformer, zip=False)

    loaded_model = cbw.load_model(model_path)
    x_test_transformed = custom_transformer(x_test)
    original_model_predictions = model.predict(x_test_transformed)
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
def test_boston_keras_data_preparation_and_preprocessing(
    preprocessor,
    add_value_to_column_transformer,
    boston_training_test,
    keras_model,
    model_path,
):
    x_train, x_test, y_train, _ = boston_training_test

    x_train_prepared = add_value_to_column_transformer(x_train)
    x_train_transformed = preprocessor.fit_transform(x_train_prepared)

    model = keras_model
    model.fit(x_train_transformed, y_train, epochs=10, batch_size=32)
    cbw.save_model(
        model_path,
        model,
        preprocessing=preprocessor,
        data_preparation=add_value_to_column_transformer,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)
    x_test_prepared = add_value_to_column_transformer(x_test)
    x_test_transformed = preprocessor.transform(x_test_prepared)
    original_model_predictions = model.predict(x_test_transformed)
    loaded_model_predictions = loaded_model.predict(x_test)

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_boston_keras_zipped_path_already_exists(
    sk_function_transformer,
    add_value_to_column_transformer,
    boston_training_test,
    keras_model,
    model_path,
):
    x_train, x_test, y_train, _ = boston_training_test

    x_train_prepared = add_value_to_column_transformer(x_train)
    x_train_transformed = sk_function_transformer.fit_transform(x_train_prepared)

    model = keras_model
    model.fit(x_train_transformed, y_train, epochs=10, batch_size=32)
    cbw.save_model(
        model_path,
        model,
        preprocessing=sk_function_transformer,
        data_preparation=add_value_to_column_transformer,
    )

    with pytest.raises(cbw.ClearboxWrapperException):
        cbw.save_model(
            model_path,
            model,
            preprocessing=sk_function_transformer,
            data_preparation=add_value_to_column_transformer,
        )


def test_boston_keras_path_already_exists(
    sk_function_transformer,
    add_value_to_column_transformer,
    boston_training_test,
    keras_model,
    model_path,
):
    x_train, x_test, y_train, _ = boston_training_test

    x_train_prepared = add_value_to_column_transformer(x_train)
    x_train_transformed = sk_function_transformer.fit_transform(x_train_prepared)

    model = keras_model
    model.fit(x_train_transformed, y_train, epochs=10, batch_size=32)
    cbw.save_model(
        model_path,
        model,
        preprocessing=sk_function_transformer,
        data_preparation=add_value_to_column_transformer,
        zip=False,
    )

    with pytest.raises(cbw.ClearboxWrapperException):
        cbw.save_model(
            model_path,
            model,
            preprocessing=sk_function_transformer,
            data_preparation=add_value_to_column_transformer,
            zip=False,
        )
