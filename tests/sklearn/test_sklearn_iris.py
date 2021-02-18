from sys import version_info

import numpy as np
import pytest
import sklearn.datasets as datasets
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.neighbors as neighbors
import sklearn.preprocessing as sk_preprocessing
import sklearn.svm as svm
import sklearn.tree as tree
import yaml

import clearbox_wrapper as cbw


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
        (linear_model.LogisticRegression(max_iter=150)),
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
    cbw.save_model(tmp_model_path, fitted_model, zip=False)
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict_proba(x[:5])
    loaded_model_predictions = loaded_model.predict_proba(x[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (
            linear_model.LogisticRegression(max_iter=150),
            sk_preprocessing.StandardScaler(),
        ),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50),
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
    cbw.save_model(tmp_model_path, fitted_model, preprocessing=preprocessor, zip=False)
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict_proba(x_transformed[:5])
    loaded_model_predictions = loaded_model.predict_proba(x[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (
            linear_model.LogisticRegression(max_iter=150),
            sk_preprocessing.StandardScaler(),
        ),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50),
        ),
        (
            neighbors.KNeighborsClassifier(),
            sk_preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal"),
        ),
        (tree.DecisionTreeClassifier(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestClassifier(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_sklearn_data_preparation_and_preprocessing_save_and_load(
    sklearn_model, preprocessor, iris_data, drop_column_transformer, tmpdir
):
    x, y = iris_data
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
    original_model_predictions = fitted_model.predict_proba(x_transformed[:5])
    loaded_model_predictions = loaded_model.predict_proba(x[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_iris_sklearn_load_preprocessing_without_preprocessing(iris_data, tmpdir):
    x, y = iris_data
    model = linear_model.LogisticRegression(max_iter=150)
    fitted_model = model.fit(x, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, zip=False)
    loaded_model = cbw.load_model(tmp_model_path)
    with pytest.raises(cbw.ClearboxWrapperException):
        loaded_model.preprocess_data(x)


def test_iris_sklearn_load_data_preparation_without_data_preparation(iris_data, tmpdir):
    x, y = iris_data
    sk_transformer = sk_preprocessing.StandardScaler()
    model = linear_model.LogisticRegression(max_iter=150)
    x_transformed = sk_transformer.fit_transform(x)
    fitted_model = model.fit(x_transformed, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(
        tmp_model_path, fitted_model, preprocessing=sk_transformer, zip=False
    )
    loaded_model = cbw.load_model(tmp_model_path)
    with pytest.raises(cbw.ClearboxWrapperException):
        loaded_model.prepare_data(x)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (
            linear_model.LogisticRegression(max_iter=150),
            sk_preprocessing.StandardScaler(),
        ),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50),
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
    cbw.save_model(tmp_model_path, fitted_model, preprocessing=preprocessor, zip=False)
    loaded_model = cbw.load_model(tmp_model_path)
    x_transformed_by_loaded_preprocessing = loaded_model.preprocess_data(x)
    np.testing.assert_array_equal(x_transformed, x_transformed_by_loaded_preprocessing)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (
            linear_model.LogisticRegression(max_iter=150),
            sk_preprocessing.StandardScaler(),
        ),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50),
        ),
        (
            neighbors.KNeighborsClassifier(),
            sk_preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal"),
        ),
        (tree.DecisionTreeClassifier(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestClassifier(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_sklearn_get_prepared_data(
    sklearn_model, preprocessor, iris_data, drop_column_transformer, tmpdir
):
    x, y = iris_data
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
        zip=False,
    )
    loaded_model = cbw.load_model(tmp_model_path)
    x_prepared_by_loaded_data_preparation = loaded_model.prepare_data(x)
    np.testing.assert_array_equal(x_prepared, x_prepared_by_loaded_data_preparation)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (
            linear_model.LogisticRegression(max_iter=150),
            sk_preprocessing.StandardScaler(),
        ),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50),
        ),
        (
            neighbors.KNeighborsClassifier(),
            sk_preprocessing.KBinsDiscretizer(n_bins=5, encode="ordinal"),
        ),
        (tree.DecisionTreeClassifier(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestClassifier(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_sklearn_get_prepared_and_processed_data(
    sklearn_model, preprocessor, iris_data, drop_column_transformer, tmpdir
):
    x, y = iris_data
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
        zip=False,
    )
    loaded_model = cbw.load_model(tmp_model_path)
    x_prepared_by_loaded_data_preparation = loaded_model.prepare_data(x)
    x_transformed_by_loaded_preprocessing = loaded_model.preprocess_data(
        x_prepared_by_loaded_data_preparation
    )
    np.testing.assert_array_equal(x_transformed, x_transformed_by_loaded_preprocessing)


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (
            linear_model.LogisticRegression(max_iter=150),
            sk_preprocessing.StandardScaler(),
        ),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50),
        ),
        (
            neighbors.KNeighborsClassifier(),
            sk_preprocessing.RobustScaler(),
        ),
        (tree.DecisionTreeClassifier(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestClassifier(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_sklearn_predict_without_preprocessing(
    sklearn_model, preprocessor, iris_data, drop_column_transformer, tmpdir
):
    x, y = iris_data
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
        zip=False,
    )
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict_proba(x_transformed[:5])
    loaded_model_predictions = loaded_model.predict_proba(x[:5], preprocess=False)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        original_model_predictions,
        loaded_model_predictions,
    )


@pytest.mark.parametrize(
    "sklearn_model, preprocessor",
    [
        (
            linear_model.LogisticRegression(max_iter=150),
            sk_preprocessing.StandardScaler(),
        ),
        (
            svm.SVC(probability=True),
            sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50),
        ),
        (
            neighbors.KNeighborsClassifier(),
            sk_preprocessing.RobustScaler(),
        ),
        (tree.DecisionTreeClassifier(), sk_preprocessing.RobustScaler()),
        (ensemble.RandomForestClassifier(), sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_iris_sklearn_predict_without_data_preparation(
    sklearn_model, preprocessor, iris_data, drop_column_transformer, tmpdir
):
    x, y = iris_data
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
        zip=False,
    )
    loaded_model = cbw.load_model(tmp_model_path)
    with pytest.raises(ValueError):
        loaded_model.predict_proba(x[:5], prepare_data=False)


@pytest.mark.parametrize(
    "sklearn_model",
    [
        (linear_model.LogisticRegression(max_iter=150)),
        (svm.SVC(probability=True)),
        (neighbors.KNeighborsClassifier()),
        (tree.DecisionTreeClassifier()),
        (ensemble.RandomForestClassifier()),
    ],
)
def test_iris_sklearn_conda_env(sklearn_model, iris_data, tmpdir):
    import sklearn
    import dill

    x, y = iris_data
    fitted_model = sklearn_model.fit(x, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, zip=False)

    with open(tmp_model_path + "/conda.yaml", "r") as f:
        conda_env = yaml.safe_load(f)

    python_version = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    sklearn_version = sklearn.__version__
    dill_version = dill.__version__

    channels_list = ["defaults", "conda-forge"]
    dependencies = [
        "python={}".format(python_version),
        "pip",
        {
            "pip": [
                "dill=={}".format(dill_version),
                "scikit-learn=={}".format(sklearn_version),
            ]
        },
    ]
    assert conda_env["channels"] == channels_list
    assert conda_env["dependencies"] == dependencies


def test_iris_sklearn_conda_env_additional_deps(iris_data, tmpdir):
    import sklearn
    import dill

    x, y = iris_data
    model = linear_model.LogisticRegression(max_iter=150)
    fitted_model = model.fit(x, y)

    add_deps = [
        "torch==1.6.0",
        "tensorflow==2.1.0",
        "fastapi==0.52.1",
        "my_package==1.23.1",
    ]

    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, additional_deps=add_deps, zip=False)

    with open(tmp_model_path + "/conda.yaml", "r") as f:
        conda_env = yaml.safe_load(f)

    python_version = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    sklearn_version = sklearn.__version__
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
                "scikit-learn=={}".format(sklearn_version),
            ]
        },
    ]
    assert conda_env["channels"] == channels_list
    assert conda_env["dependencies"] == dependencies


def test_iris_sklearn_conda_env_additional_deps_with_duplicates(iris_data, tmpdir):
    x, y = iris_data
    model = linear_model.LogisticRegression(max_iter=150)
    fitted_model = model.fit(x, y)

    add_deps = ["torch==1.6.0", "torch==1.6.2"]
    tmp_model_path = str(tmpdir + "/saved_model")
    with pytest.raises(ValueError):
        cbw.save_model(
            tmp_model_path, fitted_model, additional_deps=add_deps, zip=False
        )
