import os

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import sklearn.preprocessing as sk_preprocessing
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import clearbox_wrapper as cbw


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


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


def keras_model(input_shape):
    keras_clf = Sequential()
    keras_clf.add(Dense(27, input_dim=input_shape, activation="relu"))
    keras_clf.add(Dense(14, activation="relu"))
    keras_clf.add(Dense(7, activation="relu"))
    keras_clf.add(Dense(1, activation="sigmoid"))

    keras_clf.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return keras_clf


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

    if hasattr(pdf, "toarray"):
        pdf = pdf.toarray()
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
        # The model signature does not specify column names => we can only verify column count.
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


def test_adult_keras_preprocessing(adult_training, adult_test, model_path):

    x_training, y_training = adult_training
    x_test, _ = adult_test
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training)

    x_train_transformed = x_preprocessor.fit_transform(x_training)
    y_train_transformed = y_encoder.fit_transform(y_training)

    model = keras_model(x_train_transformed.shape[1])
    model.fit(x_train_transformed, y_train_transformed, epochs=10, batch_size=32)
    cbw.save_model(model_path, model, preprocessing=x_preprocessor, zip=False)

    loaded_model = cbw.load_model(model_path)

    x_test_transformed = x_preprocessor.transform(x_test)
    original_model_predictions = model.predict(x_test_transformed)
    loaded_model_predictions = loaded_model.predict(x_test)

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_adult_keras_preprocessing_and_data_preparation(
    adult_training, adult_test, data_preparation, model_path
):

    x_training, y_training = adult_training
    x_test, _ = adult_test
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training)

    x_training_prepared = data_preparation(x_training)
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training_prepared)

    x_train_transformed = x_preprocessor.fit_transform(x_training_prepared)
    y_train_transformed = y_encoder.fit_transform(y_training)

    model = keras_model(x_train_transformed.shape[1])
    model.fit(x_train_transformed, y_train_transformed, epochs=10, batch_size=32)
    cbw.save_model(
        model_path,
        model,
        preprocessing=x_preprocessor,
        data_preparation=data_preparation,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)

    x_test_prepared = data_preparation(x_test)
    x_test_transformed = x_preprocessor.transform(x_test_prepared)

    original_model_predictions = model.predict_proba(x_test_transformed)
    loaded_model_predictions = loaded_model.predict(x_test)

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def tests_adult_keras_zipped_path_already_exists(
    adult_training, adult_test, model_path
):

    x_training, y_training = adult_training
    x_test, _ = adult_test
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training)

    x_train_transformed = x_preprocessor.fit_transform(x_training)
    y_train_transformed = y_encoder.fit_transform(y_training)

    model = keras_model(x_train_transformed.shape[1])
    model.fit(x_train_transformed, y_train_transformed, epochs=10, batch_size=32)
    cbw.save_model(model_path, model, preprocessing=x_preprocessor)
    with pytest.raises(cbw.ClearboxWrapperException):
        cbw.save_model(model_path, model, preprocessing=x_preprocessor)


def tests_adult_keras_path_already_exists(adult_training, adult_test, model_path):

    x_training, y_training = adult_training
    x_test, _ = adult_test
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training)

    x_train_transformed = x_preprocessor.fit_transform(x_training)
    y_train_transformed = y_encoder.fit_transform(y_training)

    model = keras_model(x_train_transformed.shape[1])
    model.fit(x_train_transformed, y_train_transformed, epochs=10, batch_size=32)
    cbw.save_model(model_path, model, preprocessing=x_preprocessor, zip=False)
    with pytest.raises(cbw.ClearboxWrapperException):
        cbw.save_model(model_path, model, preprocessing=x_preprocessor, zip=False)


def test_adult_xgb_preprocessing_check_model_and_preprocessing_signature(
    adult_training, adult_test, model_path
):

    x_training, y_training = adult_training
    x_test, _ = adult_test
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training)

    x_train_transformed = x_preprocessor.fit_transform(x_training)
    y_train_transformed = y_encoder.fit_transform(y_training)

    model = keras_model(x_train_transformed.shape[1])
    model.fit(x_train_transformed, y_train_transformed, epochs=10, batch_size=32)
    cbw.save_model(
        model_path,
        model,
        preprocessing=x_preprocessor,
        input_data=x_training,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)
    original_model_predictions = model.predict(x_train_transformed[:5])
    loaded_model_predictions = loaded_model.predict(x_training[:5])
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)

    mlmodel = cbw.Model.load(model_path)
    preprocessing_input_schema = mlmodel.get_preprocessing_input_schema()
    preprocessing_output_schema = mlmodel.get_preprocessing_output_schema()
    model_input_schema = mlmodel.get_model_input_schema()

    assert _check_schema(x_training, preprocessing_input_schema)
    assert _check_schema(x_train_transformed, preprocessing_output_schema)
    assert _check_schema(x_train_transformed, model_input_schema)
    assert preprocessing_output_schema == model_input_schema


def test_adult_keras_check_model_preprocessing_and_data_preparation_signature(
    adult_training, adult_test, data_preparation, model_path
):

    x_training, y_training = adult_training
    x_test, _ = adult_test
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training)

    x_training_prepared = data_preparation(x_training)
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training_prepared)

    x_train_transformed = x_preprocessor.fit_transform(x_training_prepared)
    y_train_transformed = y_encoder.fit_transform(y_training)

    model = keras_model(x_train_transformed.shape[1])
    model.fit(x_train_transformed, y_train_transformed, epochs=10, batch_size=32)
    cbw.save_model(
        model_path,
        model,
        preprocessing=x_preprocessor,
        data_preparation=data_preparation,
        input_data=x_training,
        zip=False,
    )

    loaded_model = cbw.load_model(model_path)

    x_test_prepared = data_preparation(x_test)
    x_test_transformed = x_preprocessor.transform(x_test_prepared)

    original_model_predictions = model.predict_proba(x_test_transformed)
    loaded_model_predictions = loaded_model.predict(x_test)

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)

    mlmodel = cbw.Model.load(model_path)
    data_preparation_input_schema = mlmodel.get_data_preparation_input_schema()
    data_preparation_output_schema = mlmodel.get_data_preparation_output_schema()
    preprocessing_input_schema = mlmodel.get_preprocessing_input_schema()
    preprocessing_output_schema = mlmodel.get_preprocessing_output_schema()
    model_input_schema = mlmodel.get_model_input_schema()

    assert _check_schema(x_training, data_preparation_input_schema)
    assert _check_schema(x_training_prepared, data_preparation_output_schema)
    assert _check_schema(x_training_prepared, preprocessing_input_schema)
    assert _check_schema(x_train_transformed, preprocessing_output_schema)
    assert _check_schema(x_train_transformed, model_input_schema)
    assert not _check_schema(x_training, model_input_schema)
    assert data_preparation_output_schema == preprocessing_input_schema
    assert preprocessing_output_schema == model_input_schema
