from loguru import logger
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing as sk_preprocessing

import clearbox_wrapper.pyfunc.pyfunc
from clearbox_wrapper.signature.signature import infer_signature
import clearbox_wrapper.slearn.sklearn
import clearbox_wrapper.wrapper.wrapper


def main():
    iris = datasets.load_iris()
    iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
    clf = RandomForestClassifier(max_depth=7, random_state=0)
    clf.fit(iris_train, iris.target)
    signature = infer_signature(iris_train, clf.predict(iris_train))
    clearbox_wrapper.slearn.sklearn.save_sklearn_model(
        clf, "prova", signature=signature
    )


def main_load():
    iris = datasets.load_iris()
    iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
    loaded_model = clearbox_wrapper.pyfunc.pyfunc.load_model("prova")
    print(loaded_model)
    preds = loaded_model.predict(iris_train)
    print(preds)


def create_and_save_wrapper():
    iris = datasets.load_iris()
    iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)

    preprocessor = sk_preprocessing.StandardScaler()
    clf = RandomForestClassifier(max_depth=7, random_state=0)

    iris_train_processed = preprocessor.fit_transform(iris_train)
    clf.fit(iris_train_processed, iris.target)

    clearbox_wrapper.wrapper.wrapper.save_model(
        "prova_wrapper", clf, input_data=iris_train, preprocessing=preprocessor
    )


def load_clearbox_model(path):
    return clearbox_wrapper.wrapper.wrapper.load_model(path)


if __name__ == "__main__":
    # create_and_save_wrapper()
    loaded = load_clearbox_model("prova_wrapper")
    logger.info(type(loaded))
    iris = datasets.load_iris()
    iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
    logger.warning(dir(loaded))
    train_predictions = loaded.predict(iris_train, prepare_data=False)
    logger.info("- Original Dataset:\n{}".format(iris_train[:10]))
    logger.info(
        "- Preprocessed Dataset:\n{}".format(loaded.prepare_data(iris_train[:10]))
    )
