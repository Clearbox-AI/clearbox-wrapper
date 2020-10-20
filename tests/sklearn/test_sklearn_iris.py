import tempfile
import pytest

import numpy as np

import sklearn.datasets as datasets
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.neighbors as neighbors
import sklearn.tree as tree
import sklearn.ensemble as ensemble

import sklearn.preprocessing as preprocessing

import mlflow.pyfunc

from clearbox_wrapper.clearbox_wrapper import ClearboxWrapper


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

    transformer = preprocessing.FunctionTransformer(simple_preprocessor, validate=True)
    return transformer


@pytest.fixture()
def custom_transformer():
    def simple_preprocessor(numpy_x):
        transformed_x = numpy_x + 1.0
        return transformed_x

    return simple_preprocessor


@pytest.mark.parametrize(
    "sklearn_model, data",
    [
        (linear_model.LogisticRegression(), pytest.lazy_fixture("iris_data")),
        (svm.SVC(probability=True), pytest.lazy_fixture("iris_data")),
        (neighbors.KNeighborsClassifier(), pytest.lazy_fixture("iris_data")),
        (tree.DecisionTreeClassifier(), pytest.lazy_fixture("iris_data")),
        (ensemble.RandomForestClassifier(), pytest.lazy_fixture("iris_data")),
    ],
)
def test_iris_sklearn_no_preprocessing(sklearn_model, data):
    x, y = data
    fitted_model = sklearn_model.fit(x, y)
    wrapped_model = ClearboxWrapper(fitted_model)
    original_model_predictions = fitted_model.predict_proba(x[:5])
    wrapped_model_predictions = wrapped_model.predict(model_input=x[:5])
    np.testing.assert_array_equal(original_model_predictions, wrapped_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model, sk_transformer, data",
    [
        (
            linear_model.LogisticRegression(),
            preprocessing.StandardScaler(),
            pytest.lazy_fixture("iris_data"),
        ),
        (
            svm.SVC(probability=True),
            preprocessing.QuantileTransformer(random_state=0),
            pytest.lazy_fixture("iris_data"),
        ),
        (
            neighbors.KNeighborsClassifier(),
            preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal"),
            pytest.lazy_fixture("iris_data"),
        ),
        (
            tree.DecisionTreeClassifier(),
            preprocessing.RobustScaler(),
            pytest.lazy_fixture("iris_data"),
        ),
        (
            ensemble.RandomForestClassifier(),
            preprocessing.MaxAbsScaler(),
            pytest.lazy_fixture("iris_data"),
        ),
    ],
)
def test_iris_sklearn_preprocessing(sklearn_model, sk_transformer, data):
    x, y = data
    x_transformed = sk_transformer.fit_transform(x)
    fitted_model = sklearn_model.fit(x_transformed, y)
    wrapped_model = ClearboxWrapper(fitted_model, sk_transformer)
    original_model_predictions = fitted_model.predict_proba(
        sk_transformer.transform(x[:5])
    )
    wrapped_model_predictions = wrapped_model.predict(model_input=x[:5])
    np.testing.assert_array_equal(original_model_predictions, wrapped_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model, function_transformer, data",
    [
        (
            linear_model.LogisticRegression(),
            pytest.lazy_fixture("sk_function_transformer"),
            pytest.lazy_fixture("iris_data"),
        ),
        (
            svm.SVC(probability=True),
            pytest.lazy_fixture("sk_function_transformer"),
            pytest.lazy_fixture("iris_data"),
        ),
        (
            neighbors.KNeighborsClassifier(),
            pytest.lazy_fixture("sk_function_transformer"),
            pytest.lazy_fixture("iris_data"),
        ),
        (
            tree.DecisionTreeClassifier(),
            pytest.lazy_fixture("sk_function_transformer"),
            pytest.lazy_fixture("iris_data"),
        ),
        (
            ensemble.RandomForestClassifier(),
            pytest.lazy_fixture("sk_function_transformer"),
            pytest.lazy_fixture("iris_data"),
        ),
    ],
)
def test_iris_sklearn_preprocessing_with_function_transformer(
    sklearn_model, function_transformer, data
):
    x, y = data
    x_transformed = function_transformer.fit_transform(x)
    fitted_model = sklearn_model.fit(x_transformed, y)
    wrapped_model = ClearboxWrapper(fitted_model, function_transformer)
    original_model_predictions = fitted_model.predict_proba(x_transformed[:5])
    wrapped_model_predictions = wrapped_model.predict(model_input=x[:5])
    np.testing.assert_array_equal(original_model_predictions, wrapped_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model, custom_preprocessing, data",
    [
        (
            linear_model.LogisticRegression(),
            pytest.lazy_fixture("custom_transformer"),
            pytest.lazy_fixture("iris_data"),
        ),
        (
            svm.SVC(probability=True),
            pytest.lazy_fixture("custom_transformer"),
            pytest.lazy_fixture("iris_data"),
        ),
        (
            neighbors.KNeighborsClassifier(),
            pytest.lazy_fixture("custom_transformer"),
            pytest.lazy_fixture("iris_data"),
        ),
        (
            tree.DecisionTreeClassifier(),
            pytest.lazy_fixture("custom_transformer"),
            pytest.lazy_fixture("iris_data"),
        ),
        (
            ensemble.RandomForestClassifier(),
            pytest.lazy_fixture("custom_transformer"),
            pytest.lazy_fixture("iris_data"),
        ),
    ],
)
def test_iris_sklearn_preprocessing_with_custom_transformer(
    sklearn_model, custom_preprocessing, data
):
    x, y = data
    x_transformed = custom_preprocessing(x)
    fitted_model = sklearn_model.fit(x_transformed, y)
    wrapped_model = ClearboxWrapper(fitted_model, custom_preprocessing)
    original_model_predictions = fitted_model.predict_proba(x_transformed[:5])
    wrapped_model_predictions = wrapped_model.predict(model_input=x[:5])
    np.testing.assert_array_equal(original_model_predictions, wrapped_model_predictions)


def test_iris_sklearn_no_preprocessing_save(iris_data, tmp_path):
    x, y = iris_data
    model = linear_model.LogisticRegression()
    fitted_model = model.fit(x, y)
    wrapped_model = ClearboxWrapper(fitted_model)
    model_path = tmp_path
    print(model_path)
    wrapped_model.save(model_path)
    loaded_model = mlflow.pyfunc.load_model(model_path)
    original_model_predictions = fitted_model.predict_proba(x[:5])
    loaded_model_predictions = loaded_model.predict(x[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)