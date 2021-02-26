import numpy as np
import pandas as pd
import pytest
import sklearn.datasets as datasets
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.neighbors as neighbors
import sklearn.preprocessing as sk_preprocessing
import sklearn.svm as svm
import sklearn.tree as tree

import clearbox_wrapper as cbw


@pytest.fixture(scope="module")
def boston_data():
    boston = datasets.load_boston()
    x = boston.data
    y = boston.target
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


@pytest.mark.parametrize(
    "sklearn_model",
    [
        (linear_model.LinearRegression()),
        (svm.SVR()),
        (neighbors.KNeighborsRegressor()),
        (tree.DecisionTreeRegressor()),
        (ensemble.RandomForestRegressor()),
    ],
)
def test_boston_sklearn_no_preprocessing(sklearn_model, boston_data, tmpdir):
    x, y = boston_data
    fitted_model = sklearn_model.fit(x, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, zip=False)
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict(x[:5])
    wrapped_model_predictions = loaded_model.predict(x[:5])
    np.testing.assert_array_equal(original_model_predictions, wrapped_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (
            linear_model.LinearRegression(),
            sk_preprocessing.StandardScaler(),
        ),
        (
            svm.SVR(),
            sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50),
        ),
        (
            neighbors.KNeighborsRegressor(),
            sk_preprocessing.MaxAbsScaler(),
        ),
        (tree.DecisionTreeRegressor(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestRegressor(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_boston_sklearn_preprocessing(sklearn_model, preprocessor, boston_data, tmpdir):
    x, y = boston_data
    x_transformed = preprocessor.fit_transform(x)
    fitted_model = sklearn_model.fit(x_transformed, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, preprocessing=preprocessor, zip=False)
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict(x_transformed[:5])
    loaded_model_predictions = loaded_model.predict(x[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (
            linear_model.LinearRegression(),
            sk_preprocessing.StandardScaler(),
        ),
        (
            svm.SVR(),
            sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=20),
        ),
        (
            neighbors.KNeighborsRegressor(),
            sk_preprocessing.RobustScaler(),
        ),
        (tree.DecisionTreeRegressor(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestRegressor(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_boston_sklearn_data_preparation_and_preprocessing(
    sklearn_model, preprocessor, boston_data, drop_column_transformer, tmpdir
):
    x, y = boston_data
    data_preparation = drop_column_transformer
    x_transformed = data_preparation(x)
    x_transformed = preprocessor.fit_transform(x_transformed)
    fitted_model = sklearn_model.fit(x_transformed, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(
        tmp_model_path,
        fitted_model,
        preprocessing=preprocessor,
        data_preparation=data_preparation,
        zip=False,
    )
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict(x_transformed[:5])
    loaded_model_predictions = loaded_model.predict(x[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def tests_boston_sklearn_zipped_path_already_exists(boston_data, tmpdir):
    x, y = boston_data
    sklearn_model = tree.DecisionTreeRegressor()
    fitted_model = sklearn_model.fit(x, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model)
    with pytest.raises(cbw.ClearboxWrapperException):
        cbw.save_model(tmp_model_path, fitted_model)


def tests_boston_sklearn_path_already_exists(boston_data, tmpdir):
    x, y = boston_data
    sklearn_model = tree.DecisionTreeRegressor()
    fitted_model = sklearn_model.fit(x, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, zip=False)
    with pytest.raises(cbw.ClearboxWrapperException):
        cbw.save_model(tmp_model_path, fitted_model, zip=False)


@pytest.mark.parametrize(
    "sklearn_model",
    [
        (neighbors.KNeighborsRegressor()),
        (tree.DecisionTreeRegressor()),
    ],
)
def test_boston_sklearn_no_preprocessing_check_model_signature(
    sklearn_model, boston_data, tmpdir
):
    x, y = boston_data
    fitted_model = sklearn_model.fit(x, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, input_data=x, zip=False)
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict(x[:5])
    loaded_model_predictions = loaded_model.predict(x[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)

    mlmodel = cbw.Model.load(tmp_model_path)
    assert _check_schema(x, mlmodel.get_model_input_schema())


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (tree.DecisionTreeRegressor(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestRegressor(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_boston_sklearn_preprocessing_check_model_and_preprocessing_signature(
    sklearn_model, preprocessor, boston_data, tmpdir
):
    x, y = boston_data
    x_transformed = preprocessor.fit_transform(x)
    fitted_model = sklearn_model.fit(x_transformed, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(
        tmp_model_path,
        fitted_model,
        preprocessing=preprocessor,
        input_data=x,
        zip=False,
    )
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict(x_transformed[:5])
    loaded_model_predictions = loaded_model.predict(x[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)

    mlmodel = cbw.Model.load(tmp_model_path)
    preprocessing_input_schema = mlmodel.get_preprocessing_input_schema()
    preprocessing_output_schema = mlmodel.get_preprocessing_output_schema()
    model_input_schema = mlmodel.get_model_input_schema()

    assert _check_schema(x, preprocessing_input_schema)
    assert _check_schema(x_transformed, preprocessing_output_schema)
    assert _check_schema(x_transformed, model_input_schema)
    assert preprocessing_output_schema == model_input_schema


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (tree.DecisionTreeRegressor(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestRegressor(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_boston_sklearn_check_model_preprocessing_and_data_preparation_signature(
    sklearn_model, preprocessor, boston_data, drop_column_transformer, tmpdir
):
    x, y = boston_data
    data_preparation = drop_column_transformer
    x_prepared = data_preparation(x)
    x_transformed = preprocessor.fit_transform(x_prepared)
    fitted_model = sklearn_model.fit(x_transformed, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(
        tmp_model_path,
        fitted_model,
        preprocessing=preprocessor,
        data_preparation=data_preparation,
        input_data=x,
        zip=False,
    )
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict(x_transformed[:5])
    loaded_model_predictions = loaded_model.predict(x[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)

    mlmodel = cbw.Model.load(tmp_model_path)
    data_preparation_input_schema = mlmodel.get_data_preparation_input_schema()
    data_preparation_output_schema = mlmodel.get_data_preparation_output_schema()
    preprocessing_input_schema = mlmodel.get_preprocessing_input_schema()
    preprocessing_output_schema = mlmodel.get_preprocessing_output_schema()
    model_input_schema = mlmodel.get_model_input_schema()

    assert _check_schema(x, data_preparation_input_schema)
    assert _check_schema(x_prepared, data_preparation_output_schema)
    assert _check_schema(x_prepared, preprocessing_input_schema)
    assert _check_schema(x_transformed, preprocessing_output_schema)
    assert _check_schema(x_transformed, model_input_schema)
    assert not _check_schema(x, model_input_schema)
    assert data_preparation_output_schema == preprocessing_input_schema
    assert preprocessing_output_schema == model_input_schema
