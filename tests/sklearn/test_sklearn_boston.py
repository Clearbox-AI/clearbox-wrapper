import numpy as np
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
def test_iris_sklearn_data_preparation_and_preprocessing(
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
