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
        (linear_model.LogisticRegression(max_iter=150)),
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
        (linear_model.LogisticRegression(max_iter=200)),
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
        (linear_model.LogisticRegression(max_iter=300)),
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
    model = linear_model.LogisticRegression(max_iter=150)
    x_transformed = sk_transformer.fit_transform(x)
    fitted_model = model.fit(x_transformed, y)
    with pytest.raises(ValueError):
        cbw.ClearboxWrapper(fitted_model, data_cleaning=sk_transformer)


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
    cbw.save_model(tmp_model_path, fitted_model)
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict_proba(x[:5])
    loaded_model_predictions = loaded_model.predict(x[:5])
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
    cbw.save_model(tmp_model_path, fitted_model, preprocessor)
    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict_proba(x_transformed[:5])
    loaded_model_predictions = loaded_model.predict(x[:5])
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
    model = linear_model.LogisticRegression(max_iter=150)
    fitted_model = model.fit(x, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model)
    with pytest.raises(FileNotFoundError):
        loaded_model, preprocessing = cbw.load_model_preprocessing(tmp_model_path)


def test_iris_sklearn_load_data_cleaning_without_data_cleaning(iris_data, tmpdir):
    x, y = iris_data
    sk_transformer = sk_preprocessing.StandardScaler()
    model = linear_model.LogisticRegression(max_iter=150)
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
    cbw.save_model(tmp_model_path, fitted_model, preprocessor)
    loaded_model, loaded_preprocessing = cbw.load_model_preprocessing(tmp_model_path)
    x_transformed_by_loaded_preprocessing = loaded_preprocessing(x)
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
    import cloudpickle

    x, y = iris_data
    fitted_model = sklearn_model.fit(x, y)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model)

    with open(tmp_model_path + "/conda.yaml", "r") as f:
        conda_env = yaml.safe_load(f)

    python_version = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    sklearn_version = sklearn.__version__
    cloudpickle_version = cloudpickle.__version__

    channels_list = ["defaults", "conda-forge"]
    dependencies = [
        "python={}".format(python_version),
        "scikit-learn={}".format(sklearn_version),
        "pip",
        {"pip": ["mlflow", "cloudpickle=={}".format(cloudpickle_version)]},
    ]
    assert conda_env["channels"] == channels_list
    assert conda_env["dependencies"] == dependencies


def test_iris_sklearn_conda_env_additional_deps(iris_data, tmpdir):
    import sklearn
    import cloudpickle

    x, y = iris_data
    model = linear_model.LogisticRegression(max_iter=150)
    fitted_model = model.fit(x, y)

    conda_channels = ["special_channel", "custom_channel"]
    conda_deps = ["torch=1.6.0", "tensorflow=2.1.0"]
    pip_deps = ["fastapi==0.52.1", "my_package==1.23.1"]

    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(
        tmp_model_path,
        fitted_model,
        additional_conda_channels=conda_channels,
        additional_conda_deps=conda_deps,
        additional_pip_deps=pip_deps,
    )

    with open(tmp_model_path + "/conda.yaml", "r") as f:
        conda_env = yaml.safe_load(f)

    python_version = "{major}.{minor}.{micro}".format(
        major=version_info.major, minor=version_info.minor, micro=version_info.micro
    )
    sklearn_version = sklearn.__version__
    cloudpickle_version = cloudpickle.__version__

    channels_list = ["defaults", "conda-forge", "special_channel", "custom_channel"]
    dependencies = [
        "python={}".format(python_version),
        "torch=1.6.0",
        "tensorflow=2.1.0",
        "scikit-learn={}".format(sklearn_version),
        "pip",
        {
            "pip": [
                "mlflow",
                "cloudpickle=={}".format(cloudpickle_version),
                "fastapi==0.52.1",
                "my_package==1.23.1",
            ]
        },
    ]
    assert conda_env["channels"] == channels_list
    assert conda_env["dependencies"] == dependencies


def test_iris_sklearn_conda_env_additional_channels_with_duplicates(iris_data, tmpdir):
    x, y = iris_data
    model = linear_model.LogisticRegression(max_iter=150)
    fitted_model = model.fit(x, y)

    conda_channels = ["special_channel", "custom_channel", "custom_channel"]
    tmp_model_path = str(tmpdir + "/saved_model")
    with pytest.raises(ValueError):
        cbw.save_model(
            tmp_model_path,
            fitted_model,
            additional_conda_channels=conda_channels,
        )


def test_iris_sklearn_conda_env_additional_conda_deps_with_duplicates(
    iris_data, tmpdir
):
    x, y = iris_data
    model = linear_model.LogisticRegression(max_iter=150)
    fitted_model = model.fit(x, y)

    conda_deps = ["torch=1.6.0", "torch=1.6.2"]
    tmp_model_path = str(tmpdir + "/saved_model")
    with pytest.raises(ValueError):
        cbw.save_model(tmp_model_path, fitted_model, additional_conda_deps=conda_deps)


def test_iris_sklearn_conda_env_additional_pip_deps_with_duplicates(iris_data, tmpdir):
    x, y = iris_data
    model = linear_model.LogisticRegression(max_iter=150)
    fitted_model = model.fit(x, y)

    pip_deps = ["torch==1.6.0", "torch==1.6.2"]
    tmp_model_path = str(tmpdir + "/saved_model")
    with pytest.raises(ValueError):
        cbw.save_model(tmp_model_path, fitted_model, additional_pip_deps=pip_deps)


def test_iris_sklearn_conda_env_additional_conda_and_pip_deps_with_common_deps(
    iris_data, tmpdir
):
    x, y = iris_data
    model = linear_model.LogisticRegression(max_iter=150)
    fitted_model = model.fit(x, y)

    conda_deps = ["torch=1.6.0", "tensorflow=2.1.0"]
    pip_deps = ["torch==1.6.3", "fastapi>=0.52.1"]
    tmp_model_path = str(tmpdir + "/saved_model")
    with pytest.raises(ValueError):
        cbw.save_model(
            tmp_model_path,
            fitted_model,
            additional_conda_deps=conda_deps,
            additional_pip_deps=pip_deps,
        )
