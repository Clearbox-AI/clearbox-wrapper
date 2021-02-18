from loguru import logger
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing as sk_preprocessing

import clearbox_wrapper as cbw


def create_and_save_wrapper():
    iris = datasets.load_iris()
    iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)

    preprocessor = sk_preprocessing.StandardScaler()
    clf = RandomForestClassifier(max_depth=7, random_state=0)

    iris_train_processed = preprocessor.fit_transform(iris_train)
    clf.fit(iris_train_processed, iris.target)

    cbw.save_model(
        "prova_wrapper", clf, input_data=iris_train, preprocessing=preprocessor
    )


def create_and_save_preprocessing_data_preparation():
    def drop_column(x_dataframe):
        transformed_x = x_dataframe.drop("sepal length (cm)", axis=1)
        return transformed_x

    iris = datasets.load_iris()
    iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)

    preprocessor = sk_preprocessing.StandardScaler()
    clf = RandomForestClassifier(max_depth=7, random_state=0)
    iris_train_prepared = drop_column(iris_train)
    iris_train_processed = preprocessor.fit_transform(iris_train_prepared)
    clf.fit(iris_train_processed, iris.target)
    cbw.save_model(
        "prova_wrapper_preparation",
        clf,
        input_data=iris_train,
        preprocessing=preprocessor,
        data_preparation=drop_column,
    )


def load_clearbox_model(path):
    return cbw.load_model(path)


def load_clearbox_model_preprocessing(path):
    loaded = cbw.load_model(path)
    iris = datasets.load_iris()
    iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
    train_predictions = loaded.predict_proba(iris_train)
    logger.info("- Original Dataset:\n{}".format(iris_train[:10]))
    logger.info("- Prepared Dataset:\n{}".format(loaded.prepare_data(iris_train[:10])))
    logger.info(
        "- Preprocessed Dataset:\n{}".format(
            loaded.preprocess_data(loaded.prepare_data(iris_train[:10]))
        )
    )
    logger.info("- Predictions:\n{}".format(train_predictions))


if __name__ == "__main__":
    path = "prova_wrapper_preparation"
    # create_and_save_wrapper()
    create_and_save_preprocessing_data_preparation()
    # load_clearbox_model_preprocessing(path)
