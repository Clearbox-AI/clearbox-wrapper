from loguru import logger
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
import sklearn.ensemble as ensemble
from sklearn.impute import SimpleImputer
import sklearn.linear_model as linear_model
import sklearn.neighbors as neighbors
from sklearn.pipeline import Pipeline
import sklearn.preprocessing as sk_preprocessing
import sklearn.svm as svm
import sklearn.tree as tree

import clearbox_wrapper as cbw


@pytest.fixture(scope="module")
def adult_training():
    csv_path = "tests/datasets/adult_training_500_rows.csv"
    target_column = "income"
    adult_dataset = pd.read_csv(csv_path)
    y = adult_dataset[target_column]
    x = adult_dataset.drop(target_column, axis=1)
    return x, y


@pytest.fixture(scope="module")
def adult_test():
    csv_path = "tests/datasets/adult_test_50_rows.csv"
    target_column = "income"
    adult_dataset = pd.read_csv(csv_path)
    y = adult_dataset[target_column]
    x = adult_dataset.drop(target_column, axis=1)
    return x, y


def x_and_y_preprocessing(x_dataframe):
    ordinal_features = x_dataframe.select_dtypes(include="number").columns
    categorical_features = x_dataframe.select_dtypes(include="object").columns
    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", sk_preprocessing.StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", sk_preprocessing.OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    x_encoder = ColumnTransformer(
        transformers=[
            ("ord", ordinal_transformer, ordinal_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    y_encoder = sk_preprocessing.LabelEncoder()
    return x_encoder, y_encoder


def _check_schema(pdf, input_schema):

    logger.warning("-- input type: {}".format(type(pdf)))
    logger.warning("-- input:\n {}".format(pdf.shape))
    if hasattr(pdf, "toarray"):
        logger.debug("=====> POLLO!")
        pdf = pdf.toarray()
    if isinstance(pdf, (list, np.ndarray, dict)):
        logger.warning("-- Sono entrato nel primo IF.")
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
        logger.warning("-- input_schema.has_column_names.")
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
        logger.warning("-- input_schema.has_NOT_column_names.")
        # The model signature does not specify column names => we can only verify column count.
        logger.warning("-- pdf.columns: {}".format(pdf.columns))
        logger.warning("-- len pdf.columns: {}".format(len(pdf.columns)))
        logger.warning("-- input_schema.columns: {}".format(input_schema.columns))
        logger.warning(
            "-- len input_schema.columns: {}".format(len(input_schema.columns))
        )
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


@pytest.fixture()
def data_preparation():
    def preparation(x):
        education_map = {
            "10th": "Dropout",
            "11th": "Dropout",
            "12th": "Dropout",
            "1st-4th": "Dropout",
            "5th-6th": "Dropout",
            "7th-8th": "Dropout",
            "9th": "Dropout",
            "Preschool": "Dropout",
            "HS-grad": "High School grad",
            "Some-college": "High School grad",
            "Masters": "Masters",
            "Prof-school": "Prof-School",
            "Assoc-acdm": "Associates",
            "Assoc-voc": "Associates",
        }
        occupation_map = {
            "Adm-clerical": "Admin",
            "Armed-Forces": "Military",
            "Craft-repair": "Blue-Collar",
            "Exec-managerial": "White-Collar",
            "Farming-fishing": "Blue-Collar",
            "Handlers-cleaners": "Blue-Collar",
            "Machine-op-inspct": "Blue-Collar",
            "Other-service": "Service",
            "Priv-house-serv": "Service",
            "Prof-specialty": "Professional",
            "Protective-serv": "Other",
            "Sales": "Sales",
            "Tech-support": "Other",
            "Transport-moving": "Blue-Collar",
        }
        country_map = {
            "Cambodia": "SE-Asia",
            "Canada": "British-Commonwealth",
            "China": "China",
            "Columbia": "South-America",
            "Cuba": "Other",
            "Dominican-Republic": "Latin-America",
            "Ecuador": "South-America",
            "El-Salvador": "South-America",
            "England": "British-Commonwealth",
            "Guatemala": "Latin-America",
            "Haiti": "Latin-America",
            "Honduras": "Latin-America",
            "Hong": "China",
            "India": "British-Commonwealth",
            "Ireland": "British-Commonwealth",
            "Jamaica": "Latin-America",
            "Laos": "SE-Asia",
            "Mexico": "Latin-America",
            "Nicaragua": "Latin-America",
            "Outlying-US(Guam-USVI-etc)": "Latin-America",
            "Peru": "South-America",
            "Philippines": "SE-Asia",
            "Puerto-Rico": "Latin-America",
            "Scotland": "British-Commonwealth",
            "Taiwan": "China",
            "Thailand": "SE-Asia",
            "Trinadad&Tobago": "Latin-America",
            "United-States": "United-States",
            "Vietnam": "SE-Asia",
        }
        married_map = {
            "Never-married": "Never-Married",
            "Married-AF-spouse": "Married",
            "Married-civ-spouse": "Married",
            "Married-spouse-absent": "Separated",
            "Divorced": "Separated",
        }
        mapping = {
            "education": education_map,
            "occupation": occupation_map,
            "native_country": country_map,
            "marital_status": married_map,
        }
        transformed_x = x.replace(mapping)
        return transformed_x

    return preparation


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
def test_adult_sklearn_preprocessing(sklearn_model, adult_training, adult_test, tmpdir):
    x_training, y_training = adult_training
    x_test, y_test = adult_test
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training)

    x_transformed = x_preprocessor.fit_transform(x_training)
    y_transformed = y_encoder.fit_transform(y_training)

    fitted_model = sklearn_model.fit(x_transformed, y_transformed)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(
        tmp_model_path, fitted_model, preprocessing=x_preprocessor, zip=False
    )

    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict_proba(
        x_preprocessor.transform(x_test)
    )
    loaded_model_predictions = loaded_model.predict_proba(x_test)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


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
def test_adult_sklearn_preprocessing_and_data_preparation(
    sklearn_model, adult_training, adult_test, data_preparation, tmpdir
):
    x_training, y_training = adult_training
    x_test, y_test = adult_test

    x_training_prepared = data_preparation(x_training)
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training_prepared)

    x_transformed = x_preprocessor.fit_transform(x_training_prepared)
    y_transformed = y_encoder.fit_transform(y_training)

    fitted_model = sklearn_model.fit(x_transformed, y_transformed)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(
        tmp_model_path,
        fitted_model,
        preprocessing=x_preprocessor,
        data_preparation=data_preparation,
        zip=False,
    )

    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict(
        x_preprocessor.transform(data_preparation(x_test))
    )
    loaded_model_predictions = loaded_model.predict(x_test)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_adult_sklearn_get_prepared_data(
    adult_training, adult_test, data_preparation, tmpdir
):
    x_training, y_training = adult_training
    x_test, y_test = adult_test

    x_training_prepared = data_preparation(x_training)
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training_prepared)

    x_transformed = x_preprocessor.fit_transform(x_training_prepared)
    y_transformed = y_encoder.fit_transform(y_training)

    fitted_model = tree.DecisionTreeClassifier().fit(x_transformed, y_transformed)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(
        tmp_model_path,
        fitted_model,
        preprocessing=x_preprocessor,
        data_preparation=data_preparation,
        zip=False,
    )

    loaded_model = cbw.load_model(tmp_model_path)

    x_test_prepared_by_loaded_data_preparation = loaded_model.prepare_data(x_test)
    x_test_prepared = data_preparation(x_test)

    pd.testing.assert_frame_equal(
        x_test_prepared, x_test_prepared_by_loaded_data_preparation
    )


def test_adult_sklearn_get_preprocessed_data(
    adult_training, adult_test, data_preparation, tmpdir
):
    x_training, y_training = adult_training
    x_test, y_test = adult_test

    x_training_prepared = data_preparation(x_training)
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training_prepared)

    x_transformed = x_preprocessor.fit_transform(x_training_prepared)
    y_transformed = y_encoder.fit_transform(y_training)

    fitted_model = tree.DecisionTreeClassifier().fit(x_transformed, y_transformed)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(
        tmp_model_path,
        fitted_model,
        preprocessing=x_preprocessor,
        data_preparation=data_preparation,
        zip=False,
    )

    loaded_model = cbw.load_model(tmp_model_path)

    x_test_prepared_by_loaded_data_preparation = loaded_model.prepare_data(x_test)
    x_test_transformed_by_loaded_preprocessing = loaded_model.preprocess_data(
        x_test_prepared_by_loaded_data_preparation
    )

    x_test_prepared = data_preparation(x_test)
    x_test_transformed = x_preprocessor.transform(x_test_prepared)

    np.testing.assert_array_equal(
        x_test_transformed.todense(),
        x_test_transformed_by_loaded_preprocessing.todense(),
    )


def tests_adult_sklearn_zipped_path_already_exists(adult_training, adult_test, tmpdir):
    x_training, y_training = adult_training
    x_test, y_test = adult_test
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training)

    x_transformed = x_preprocessor.fit_transform(x_training)
    y_transformed = y_encoder.fit_transform(y_training)

    sklearn_model = tree.DecisionTreeClassifier()
    fitted_model = sklearn_model.fit(x_transformed, y_transformed)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, preprocessing=x_preprocessor)
    with pytest.raises(cbw.ClearboxWrapperException):
        cbw.save_model(tmp_model_path, fitted_model)


def tests_adult_sklearn_path_already_exists(adult_training, adult_test, tmpdir):
    x_training, y_training = adult_training
    x_test, y_test = adult_test
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training)

    x_transformed = x_preprocessor.fit_transform(x_training)
    y_transformed = y_encoder.fit_transform(y_training)

    sklearn_model = tree.DecisionTreeClassifier()
    fitted_model = sklearn_model.fit(x_transformed, y_transformed)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(
        tmp_model_path, fitted_model, preprocessing=x_preprocessor, zip=False
    )
    with pytest.raises(cbw.ClearboxWrapperException):
        cbw.save_model(tmp_model_path, fitted_model, zip=False)


@pytest.mark.parametrize(
    "sklearn_model",
    [
        (tree.DecisionTreeClassifier()),
        (ensemble.RandomForestClassifier()),
    ],
)
def test_adult_sklearn_preprocessing_check_model_and_preprocessing_signature(
    sklearn_model, adult_training, adult_test, tmpdir
):
    x_training, y_training = adult_training
    x_test, y_test = adult_test
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training)

    x_transformed = x_preprocessor.fit_transform(x_training)
    y_transformed = y_encoder.fit_transform(y_training)

    sklearn_model = tree.DecisionTreeClassifier()
    fitted_model = sklearn_model.fit(x_transformed, y_transformed)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(
        tmp_model_path,
        fitted_model,
        preprocessing=x_preprocessor,
        input_data=x_training,
        zip=False,
    )

    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict(x_transformed[:5])
    loaded_model_predictions = loaded_model.predict(x_training[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)

    mlmodel = cbw.Model.load(tmp_model_path)
    preprocessing_input_schema = mlmodel.get_preprocessing_input_schema()
    preprocessing_output_schema = mlmodel.get_preprocessing_output_schema()
    model_input_schema = mlmodel.get_model_input_schema()

    logger.warning(preprocessing_input_schema)
    logger.warning(preprocessing_output_schema)

    assert _check_schema(x_training, preprocessing_input_schema)
    assert _check_schema(x_transformed, preprocessing_output_schema)
    assert _check_schema(x_transformed, model_input_schema)
    assert preprocessing_output_schema == model_input_schema


@pytest.mark.parametrize(
    "sklearn_model",
    [
        (tree.DecisionTreeClassifier()),
        (ensemble.RandomForestClassifier()),
    ],
)
def test_adult_sklearn_check_model_preprocessing_and_data_preparation_signature(
    sklearn_model, adult_training, adult_test, data_preparation, tmpdir
):
    x_training, y_training = adult_training
    x_test, y_test = adult_test

    x_training_prepared = data_preparation(x_training)
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training_prepared)

    x_transformed = x_preprocessor.fit_transform(x_training_prepared)
    y_transformed = y_encoder.fit_transform(y_training)

    fitted_model = sklearn_model.fit(x_transformed, y_transformed)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(
        tmp_model_path,
        fitted_model,
        preprocessing=x_preprocessor,
        data_preparation=data_preparation,
        input_data=x_training,
        zip=False,
    )

    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict(
        x_preprocessor.transform(data_preparation(x_test))
    )
    loaded_model_predictions = loaded_model.predict(x_test)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)

    mlmodel = cbw.Model.load(tmp_model_path)
    data_preparation_input_schema = mlmodel.get_data_preparation_input_schema()
    data_preparation_output_schema = mlmodel.get_data_preparation_output_schema()
    preprocessing_input_schema = mlmodel.get_preprocessing_input_schema()
    preprocessing_output_schema = mlmodel.get_preprocessing_output_schema()
    model_input_schema = mlmodel.get_model_input_schema()

    assert _check_schema(x_training, data_preparation_input_schema)
    assert _check_schema(x_training_prepared, data_preparation_output_schema)
    assert _check_schema(x_training_prepared, preprocessing_input_schema)
    assert _check_schema(x_transformed, preprocessing_output_schema)
    assert _check_schema(x_transformed, model_input_schema)
    assert not _check_schema(x_training, model_input_schema)
    assert data_preparation_output_schema == preprocessing_input_schema
    assert preprocessing_output_schema == model_input_schema
