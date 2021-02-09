import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

import clearbox_wrapper.pyfunc.pyfunc
import clearbox_wrapper.slearn.sklearn


def main():
    iris = datasets.load_iris()
    iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
    clf = RandomForestClassifier(max_depth=7, random_state=0)
    clf.fit(iris_train, iris.target)
    clearbox_wrapper.slearn.sklearn.save_model(clf, "prova")


def main_load():
    iris = datasets.load_iris()
    iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
    loaded_model = clearbox_wrapper.pyfunc.pyfunc.load_model("prova")
    print(loaded_model)
    preds = loaded_model.predict(iris_train)
    print(preds)


if __name__ == "__main__":
    main_load()
