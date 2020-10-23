import pytest

import numpy as np

import sklearn.datasets as datasets
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.neighbors as neighbors
import sklearn.tree as tree
import sklearn.ensemble as ensemble

import sklearn.preprocessing as sk_preprocessing

import clearbox_wrapper.clearbox_wrapper as cbw


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


@pytest.mark.parametrize(
    "sklearn_model",
    [
        (linear_model.LogisticRegression()),
        (svm.SVC(probability=True)),
        (neighbors.KNeighborsClassifier()),
        (tree.DecisionTreeClassifier()),
        (ensemble.RandomForestClassifier()),
    ],
)
def test_iris_sklearn_no_preprocessing(sklearn_model, iris_data):
    x, y = iris_data
    fitted_model = sklearn_model.fit(x, y)
    wrapped_model = cbw.ClearboxWrapper(fitted_model)
    original_model_predictions = fitted_model.predict_proba(x[:5])
    wrapped_model_predictions = wrapped_model.predict(model_input=x[:5])
    np.testing.assert_array_equal(original_model_predictions, wrapped_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model, sk_transformer",
    [
        (linear_model.LogisticRegression(), sk_preprocessing.StandardScaler()),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0),
        ),
        (
            neighbors.KNeighborsClassifier(),
            sk_preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal"),
        ),
        (tree.DecisionTreeClassifier(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestClassifier(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_sklearn_preprocessing(sklearn_model, sk_transformer, iris_data):
    x, y = iris_data
    x_transformed = sk_transformer.fit_transform(x)
    fitted_model = sklearn_model.fit(x_transformed, y)
    wrapped_model = cbw.ClearboxWrapper(fitted_model, sk_transformer)
    original_model_predictions = fitted_model.predict_proba(
        sk_transformer.transform(x[:5])
    )
    wrapped_model_predictions = wrapped_model.predict(model_input=x[:5])
    np.testing.assert_array_equal(original_model_predictions, wrapped_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model",
    [
        (linear_model.LogisticRegression()),
        (svm.SVC(probability=True)),
        (neighbors.KNeighborsClassifier()),
        (tree.DecisionTreeClassifier()),
        (ensemble.RandomForestClassifier()),
    ],
)
def test_iris_sklearn_preprocessing_with_function_transformer(
    sklearn_model, sk_function_transformer, iris_data
):
    x, y = iris_data
    x_transformed = sk_function_transformer.fit_transform(x)
    fitted_model = sklearn_model.fit(x_transformed, y)
    wrapped_model = cbw.ClearboxWrapper(fitted_model, sk_function_transformer)
    original_model_predictions = fitted_model.predict_proba(x_transformed[:5])
    wrapped_model_predictions = wrapped_model.predict(model_input=x[:5])
    np.testing.assert_array_equal(original_model_predictions, wrapped_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model",
    [
        (linear_model.LogisticRegression()),
        (svm.SVC(probability=True)),
        (neighbors.KNeighborsClassifier()),
        (tree.DecisionTreeClassifier()),
        (ensemble.RandomForestClassifier()),
    ],
)
def test_iris_sklearn_preprocessing_with_custom_transformer(
    sklearn_model, custom_transformer, iris_data
):
    x, y = iris_data
    x_transformed = custom_transformer(x)
    fitted_model = sklearn_model.fit(x_transformed, y)
    wrapped_model = cbw.ClearboxWrapper(fitted_model, custom_transformer)
    original_model_predictions = fitted_model.predict_proba(x_transformed[:5])
    wrapped_model_predictions = wrapped_model.predict(model_input=x[:5])
    np.testing.assert_array_equal(original_model_predictions, wrapped_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (linear_model.LogisticRegression(), sk_preprocessing.StandardScaler()),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0),
        ),
        (
            neighbors.KNeighborsClassifier(),
            sk_preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal"),
        ),
        (tree.DecisionTreeClassifier(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestClassifier(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_sklearn_data_cleaning_and_preprocessing(
    sklearn_model, preprocessor, iris_data, drop_column_transformer, tmpdir
):
    x, y = iris_data
    data_cleaning = drop_column_transformer
    x_transformed = data_cleaning(x)
    x_transformed = preprocessor.fit_transform(x_transformed)
    fitted_model = sklearn_model.fit(x_transformed, y)
    wrapped_model = cbw.ClearboxWrapper(fitted_model, preprocessor, data_cleaning)
    original_model_predictions = fitted_model.predict_proba(x_transformed[:5])
    wrapped_model_predictions = wrapped_model.predict(model_input=x[:5])
    np.testing.assert_array_equal(original_model_predictions, wrapped_model_predictions)


def test_iris_sklearn_data_cleaning_without_preprocessing(iris_data):
    x, y = iris_data
    sk_transformer = sk_preprocessing.StandardScaler()
    model = linear_model.LogisticRegression()
    x_transformed = sk_transformer.fit_transform(x)
    fitted_model = model.fit(x_transformed, y)
    with pytest.raises(ValueError):
        cbw.ClearboxWrapper(fitted_model, data_cleaning=sk_transformer)


@pytest.mark.parametrize(
    "sklearn_model",
    [
        (linear_model.LogisticRegression()),
        (svm.SVC(probability=True)),
        (neighbors.KNeighborsClassifier()),
        (tree.DecisionTreeClassifier()),
        (ensemble.RandomForestClassifier()),
    ],
)
def test_iris_sklearn_no_preprocessing_save_and_load(sklearn_model, iris_data, tmpdir):
    x, y = iris_data
    fitted_model = sklearn_model.fit(x, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model)
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict_proba(x[:5])
    loaded_model_predictions = loaded_model.predict(x[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (linear_model.LogisticRegression(), sk_preprocessing.StandardScaler()),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0),
        ),
        (
            neighbors.KNeighborsClassifier(),
            sk_preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal"),
        ),
        (tree.DecisionTreeClassifier(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestClassifier(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_sklearn_preprocessing_save_and_load(
    sklearn_model, preprocessor, iris_data, tmpdir
):
    x, y = iris_data
    x_transformed = preprocessor.fit_transform(x)
    fitted_model = sklearn_model.fit(x_transformed, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, preprocessor)
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict_proba(x_transformed[:5])
    loaded_model_predictions = loaded_model.predict(x[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (linear_model.LogisticRegression(), sk_preprocessing.StandardScaler()),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0),
        ),
        (
            neighbors.KNeighborsClassifier(),
            sk_preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal"),
        ),
        (tree.DecisionTreeClassifier(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestClassifier(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_sklearn_data_cleaning_and_preprocessing_save_and_load(
    sklearn_model, preprocessor, iris_data, drop_column_transformer, tmpdir
):
    x, y = iris_data
    data_cleaning = drop_column_transformer
    x_transformed = data_cleaning(x)
    x_transformed = preprocessor.fit_transform(x_transformed)
    fitted_model = sklearn_model.fit(x_transformed, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, preprocessor, data_cleaning)
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict_proba(x_transformed[:5])
    loaded_model_predictions = loaded_model.predict(x[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_sklearn_load_preprocessing_without_preprocessing(iris_data, tmpdir):
    x, y = iris_data
    model = linear_model.LogisticRegression()
    fitted_model = model.fit(x, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model)
    with pytest.raises(FileNotFoundError):
        loaded_model, preprocessing = cbw.load_model_preprocessing(tmp_model_path)


def test_iris_sklearn_load_data_cleaning_without_data_cleaning(iris_data, tmpdir):
    x, y = iris_data
    sk_transformer = sk_preprocessing.StandardScaler()
    model = linear_model.LogisticRegression()
    x_transformed = sk_transformer.fit_transform(x)
    fitted_model = model.fit(x_transformed, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, sk_transformer)
    with pytest.raises(FileNotFoundError):
        (
            loaded_model,
            preprocessing,
            data_cleaning,
        ) = cbw.load_model_preprocessing_data_cleaning(tmp_model_path)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (linear_model.LogisticRegression(), sk_preprocessing.StandardScaler()),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0),
        ),
        (
            neighbors.KNeighborsClassifier(),
            sk_preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal"),
        ),
        (tree.DecisionTreeClassifier(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestClassifier(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_sklearn_get_preprocessed_data(
    sklearn_model, preprocessor, iris_data, tmpdir
):
    x, y = iris_data
    x_transformed = preprocessor.fit_transform(x)
    fitted_model = sklearn_model.fit(x_transformed, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, preprocessor)
    loaded_model, loaded_preprocessing = cbw.load_model_preprocessing(tmp_model_path)
    x_transformed_by_loaded_preprocessing = loaded_preprocessing(x)
    np.testing.assert_array_equal(x_transformed, x_transformed_by_loaded_preprocessing)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (linear_model.LogisticRegression(), sk_preprocessing.StandardScaler()),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0),
        ),
        (
            neighbors.KNeighborsClassifier(),
            sk_preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal"),
        ),
        (tree.DecisionTreeClassifier(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestClassifier(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_sklearn_get_cleaned_data(
    sklearn_model, preprocessor, iris_data, drop_column_transformer, tmpdir
):
    x, y = iris_data
    data_cleaner = drop_column_transformer
    x_cleaned = data_cleaner(x)
    x_transformed = preprocessor.fit_transform(x_cleaned)
    fitted_model = sklearn_model.fit(x_transformed, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, preprocessor, data_cleaner)
    (
        loaded_model,
        loaded_preprocessing,
        loaded_data_cleaning,
    ) = cbw.load_model_preprocessing_data_cleaning(tmp_model_path)
    x_cleaned_by_loaded_data_cleaning = loaded_data_cleaning(x)
    np.testing.assert_array_equal(x_cleaned, x_cleaned_by_loaded_data_cleaning)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (linear_model.LogisticRegression(), sk_preprocessing.StandardScaler()),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0),
        ),
        (
            neighbors.KNeighborsClassifier(),
            sk_preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal"),
        ),
        (tree.DecisionTreeClassifier(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestClassifier(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_sklearn_get_cleaned_and_processed_data(
    sklearn_model, preprocessor, iris_data, drop_column_transformer, tmpdir
):
    x, y = iris_data
    data_cleaner = drop_column_transformer
    x_cleaned = data_cleaner(x)
    x_transformed = preprocessor.fit_transform(x_cleaned)
    fitted_model = sklearn_model.fit(x_transformed, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, preprocessor, data_cleaner)
    (
        loaded_model,
        loaded_preprocessing,
        loaded_data_cleaning,
    ) = cbw.load_model_preprocessing_data_cleaning(tmp_model_path)
    x_cleaned_by_loaded_data_cleaning = loaded_data_cleaning(x)
    x_transformed_by_loaded_preprocessing = loaded_preprocessing(
        x_cleaned_by_loaded_data_cleaning
    )
    np.testing.assert_array_equal(x_transformed, x_transformed_by_loaded_preprocessing)


def test_iris_sklearn_conda_env(iris_data):
    x, y = iris_data
    model = linear_model.LogisticRegression()
    fitted_model = model.fit(x, y)
    cbw.save_model('tests/prova_conda', fitted_model)
    assert False
