import pytest

import pandas as pd
import numpy as np

import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.neighbors as neighbors
import sklearn.tree as tree
import sklearn.ensemble as ensemble

import sklearn.preprocessing as sk_preprocessing

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

import clearbox_wrapper.clearbox_wrapper as cbw


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


@pytest.fixture()
def data_cleaning():
    def cleaning(x):
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

    return cleaning


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
    cbw.save_model(tmp_model_path, fitted_model, x_preprocessor)

    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict_proba(
        x_preprocessor.transform(x_test)
    )
    loaded_model_predictions = loaded_model.predict(x_test)
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
def test_adult_sklearn_preprocessing_and_data_cleaning(
    sklearn_model, adult_training, adult_test, data_cleaning, tmpdir
):
    x_training, y_training = adult_training
    x_test, y_test = adult_test

    x_training_cleaned = data_cleaning(x_training)
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training_cleaned)

    x_transformed = x_preprocessor.fit_transform(x_training_cleaned)
    y_transformed = y_encoder.fit_transform(y_training)

    fitted_model = sklearn_model.fit(x_transformed, y_transformed)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, x_preprocessor, data_cleaning)

    loaded_model = cbw.load_model(tmp_model_path)
    original_model_predictions = fitted_model.predict_proba(
        x_preprocessor.transform(data_cleaning(x_test))
    )
    loaded_model_predictions = loaded_model.predict(x_test)
    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_adult_sklearn_get_cleaned_data(
    adult_training, adult_test, data_cleaning, tmpdir
):
    x_training, y_training = adult_training
    x_test, y_test = adult_test

    x_training_cleaned = data_cleaning(x_training)
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training_cleaned)

    x_transformed = x_preprocessor.fit_transform(x_training_cleaned)
    y_transformed = y_encoder.fit_transform(y_training)

    fitted_model = tree.DecisionTreeClassifier().fit(x_transformed, y_transformed)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, x_preprocessor, data_cleaning)

    (
        loaded_model,
        loaded_preprocessing,
        loaded_data_cleaning,
    ) = cbw.load_model_preprocessing_data_cleaning(tmp_model_path)

    x_test_cleaned_by_loaded_data_cleaning = loaded_data_cleaning(x_test)
    x_test_cleaned = data_cleaning(x_test)

    pd.testing.assert_frame_equal(
        x_test_cleaned, x_test_cleaned_by_loaded_data_cleaning
    )


def test_adult_sklearn_get_preprocessed_data(
    adult_training, adult_test, data_cleaning, tmpdir
):
    x_training, y_training = adult_training
    x_test, y_test = adult_test

    x_training_cleaned = data_cleaning(x_training)
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training_cleaned)

    x_transformed = x_preprocessor.fit_transform(x_training_cleaned)
    y_transformed = y_encoder.fit_transform(y_training)

    fitted_model = tree.DecisionTreeClassifier().fit(x_transformed, y_transformed)
    tmp_model_path = str(tmpdir + "/saved_model")
    cbw.save_model(tmp_model_path, fitted_model, x_preprocessor, data_cleaning)

    (
        loaded_model,
        loaded_preprocessing,
        loaded_data_cleaning,
    ) = cbw.load_model_preprocessing_data_cleaning(tmp_model_path)

    x_test_cleaned_by_loaded_data_cleaning = loaded_data_cleaning(x_test)
    x_test_transformed_by_loaded_preprocessing = loaded_preprocessing(
        x_test_cleaned_by_loaded_data_cleaning
    )

    x_test_cleaned = data_cleaning(x_test)
    x_test_transformed = x_preprocessor.transform(x_test_cleaned)

    np.testing.assert_array_equal(
        x_test_transformed.todense(),
        x_test_transformed_by_loaded_preprocessing.todense(),
    )
