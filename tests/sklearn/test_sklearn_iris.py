import pytest

import numpy as np

import sklearn.datasets as datasets
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.neighbors as neighbors

from clearbox_wrapper.clearbox_wrapper import ClearboxWrapper


@pytest.fixture(scope="module")
def iris_data():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    return x, y


@pytest.fixture(scope="module")
def sklearn_log_reg_model(iris_data):
    x, y = iris_data
    log_reg_model = linear_model.LogisticRegression()
    log_reg_model.fit(x, y)
    return log_reg_model


@pytest.fixture(scope="module")
def sklearn_svm_model(iris_data):
    x, y = iris_data
    svm_model = svm.SVC()
    svm_model.fit(x, y)
    return svm_model


@pytest.fixture(scope="module")
def sklearn_knn_model(iris_data):
    x, y = iris_data
    knn_model = neighbors.KNeighborsClassifier()
    knn_model.fit(x, y)
    return knn_model


@pytest.mark.parametrize(
    "sklearn_model, data",
    [(pytest.lazy_fixture("sklearn_log_reg_model"), pytest.lazy_fixture("iris_data")),
    (pytest.lazy_fixture("sklearn_svm_model"), pytest.lazy_fixture("iris_data")),
    (pytest.lazy_fixture("sklearn_knn_model"), pytest.lazy_fixture("iris_data"))],
)
def test_iris_sklearn_no_preprocessing(sklearn_model, data):
    x, y = data
    wrapped_model = ClearboxWrapper(sklearn_model)
    original_model_predictions = sklearn_model.predict(x[:5])
    wrapped_model_predictions = wrapped_model.predict(x[:5])
    np.testing.assert_array_equal(original_model_predictions, wrapped_model_predictions)
