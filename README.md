[![Tests](https://github.com/Clearbox-AI/clearbox-wrapper/workflows/Tests/badge.svg)](https://github.com/Clearbox-AI/clearbox-wrapper/actions?workflow=Tests)

[![PyPI](https://img.shields.io/pypi/v/clearbox-wrapper.svg)](https://pypi.org/project/clearbox-wrapper/)

# Clearbox AI Wrapper

Clearbox AI Wrapper is a Python library to package and save a Machine Learning model built with common ML/DL frameworks. It is designed to wrap models trained on strutured data. It includes optional **preprocessing** and **data preparation** functions which can be used to build ready-to-production pipelines.

## Main Features

The wrapper was born as a fork from [mlflow](https://github.com/mlflow/mlflow) and it's based on its [standard format](https://mlflow.org/docs/latest/models.html). It adds the possibility to package, together with the fitted model, preprocessing and data preparation functions in order to create a production-ready pipeline able to receive new data, preprocess them and makes predictions. The resulting wrapped model/pipeline is saved as a zipped folder.

The library is designed to automatically detect the model framework and its version adding this information to the requirements saved into the final folder. Additional dependencies (e.g. libraries used in preprocessing or data preparation) can also be added as a list parameter if necessary.

The resulting wrapped folder can be loaded via the Wrapper and the model will be ready to take input through the `predict` or `predict_proba` (if present) method.

**IMPORTANT**: Currently, it is necessary to load the wrapped model with the same Python version with which the model was saved.

## No preprocessing

In the simplest case, the original dataset has already been preprocessed or it doesn't need any preprocessing. It contains only numerical values (ordinal features or one-hot encoded categorical features) and we can easily train a model on it. Then, we only need to save the model and it will be ready to receive new data to make predictions on it.

![](/docs/images/clearbox_ai_wrapper_no_preprocessing.png)

The following lines show how to wrap and save a simple Scikit-Learn model without preprocessing or data preparation:

```python
import clearbox_wrapper as cbw

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
cbw.save_model('wrapped_model_path', model)
```

## Preprocessing

Typically, data are preprocessed before being fed into the model. It is almost always necessary to transform (e.g. scaling, binarizing,...) raw data values into a representation that is more suitable for the downstream model. Most kinds of ML models take only numeric data as input, so we must at least encode the non-numeric data, if any.

Preprocessing is usually written and performed separately, before building and training the model. We fit some transformers, transform the whole dataset(s) and train the model on the processed data. If the model goes into production, we need to ship the preprocessing as well. New raw data must be processed on the same way the training dataset was.

With Clearbox AI Wrapper it's possible to wrap and save the preprocessing along with the model so to have a pipeline Processing+Model ready to take raw data, pre-process them and make predictions.

![](/docs/images/clearbox_ai_wrapper_preprocessing.png)

All the preprocessing code **must be wrapped in a single function** so it can be passed as the `preprocessing` parameter to the `save_model` method. You can use your own custom code for the preprocessing, just remember to wrap all of it in a single function, save it along with the model and add any extra dependencies.

**IMPORTANT**: If the preprocessing includes any kind of fitting on the training dataset (e.g. Scikit Learn transformers), it must be performed outside the final preprocessing function to save. Fit the transformer(s) outside the function and put only the `transform` method inside it. Furthermore, if the entire preprocessing is performed with a single Scikit-Learn transformer, you can directly pass it (fitted) to the `save_model` method.

```python
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

import clearbox_wrapper as cbw


x, y = dataset
x_preprocessor = RobustScaler()
x_preprocessed = x_preprocessor.fit_transform(x)

model = xgb.XGBClassifier(use_label_encoder=False)
fitted_model = model.fit(x_preprocessed, y)
cbw.save_model('wrapped_model_path', fitted_model, preprocessing=x_preprocessor)
```

## Data Preparation (advanced usage)

For a complex task, a single-step preprocessing could be not enough. Raw data initially collected could be very noisy, contain useless columns or splitted into different dataframes/tables sources. A first data processing is usually performed even before considering any kind of model to feed the data in. The entire dataset is cleaned and the following additional processing and the model are built considering only the cleaned data. But this is not always the case. Sometimes, this situation still applies for data fed in real time to a model in production.

We believe that a two-step data processing is required to deal with this situation. We refer to the first additional step by the term **Data Preparation**. With Clearbox AI Wrapper it's possible to wrap a data preparation step as well, in order to save a final Data Preparation + Preprocessing + Model pipeline ready to takes input.

![](/docs/images/clearbox_ai_wrapper_preprocessing_data_preparation.png)

All the data preparation code **must be wrapped in a single function** so it can be passed as the `data_preparation` parameter to the `save_model` method. The same considerations wrote above for the preprocessing step still apply for data preparation.

```python
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import clearbox_wrapper as cbw

def preparation(x):
    data_prepared = np.delete(x, 0, axis=1)
    return data_prepared

x_preprocessor = RobustScaler()

x, y = dataset
x_prepared = preparation(x)
x_preprocessed = x_preprocessor.fit_transform(x_prepared)

model = Sequential()
model.add(Dense(8, input_dim=x_preprocessed.shape[1], activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(x_preprocessed, y)

cbw.save_model(
    'wrapped_model_path',
    model,
    preprocessing=x_preprocessor,
    data_preparation=preparation
)
```

### Data Preparation vs. Preprocessing

It is not always clear which are the differences between preprocessing and data preparation. It's not easy to understand where data preparation ends and preprocessing begins. There are no conditions that apply in any case, but in general you should build the data preparation step working only with the dataset, without considering the model your data will be fed into. Any kind of operation is allowed, but often preparing the raw data includes removing or normalizing some columns, replacing values, add a column based on other column values,... After this step, no matter what kind of transformation the data have been through, they should still be readable and understandable by a human user.

The preprocessing step, on the contrary, should be considered closely tied with the downstream ML model and adapted to its particular "needs". Typically processed data by this second step are only numeric and non necessarily understandable by a human.

## Supported ML frameworks

- Scikit-Learn
- XGBoost
- Keras
- Pytorch

## Installation

Install the latest relased version on the [Python Package Index (PyPI)](https://pypi.org/project/clearbox-wrapper/) with

```shell
pip install clearbox-wrapper
```
## Examples

The following Jupyter notebooks provide examples of simle and complex cases:

- [Scikit Learn Decision Tree on Iris Dataset](https://github.com/Clearbox-AI/clearbox-wrapper/blob/master/examples/1_iris_sklearn/1_Clearbox_Wrapper_Iris_Scikit.ipynb) (No preprocessing, No data preparation)
- [XGBoost Model on Lending Club Loans Dataset](https://github.com/Clearbox-AI/clearbox-wrapper/blob/master/examples/2_loans_preprocessing_xgboost/2_Clearbox_Wrapper_Loans_Xgboost.ipynb) (Preprocessing, No data preparation)
- [Pytorch Network on Boston Housing Dataset](https://github.com/Clearbox-AI/clearbox-wrapper/blob/master/examples/3_boston_preprocessing_pytorch/3_Clearbox_Wrapper_Boston_Pytorch.ipynb) (Preprocessing, No data preparation)
- [Keras Network on UCI Adult Dataset](https://github.com/Clearbox-AI/clearbox-wrapper/blob/master/examples/4_adult_data_cleaning_preprocessing_keras/4_Clearbox_Wrapper_Adult_Keras.ipynb) (Preprocessing and data preparation)
- [Pytorch Network on Diabetes Hospital Readmissions](https://github.com/Clearbox-AI/clearbox-wrapper/blob/master/examples/5_hospital_preprocessing_pytorch/5_Clearbox_Wrapper_Hospital_Pytorch.ipynb) (Preprocessing and data preparation)

## License

[Apache License 2.0](https://github.com/Clearbox-AI/clearbox-wrapper/blob/master/LICENSE)
