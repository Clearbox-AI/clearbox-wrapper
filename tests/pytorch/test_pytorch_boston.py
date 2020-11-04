import os

import pytest

import pandas as pd
import numpy as np

import sklearn.preprocessing as sk_preprocessing
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

import clearbox_wrapper.clearbox_wrapper as cbw

num_epochs = 20
learning_rate = 0.0001
size_hidden1 = 25
size_hidden2 = 12
size_hidden3 = 6
size_hidden4 = 1


class BostonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(13, size_hidden1)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(size_hidden1, size_hidden2)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(size_hidden2, size_hidden3)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(size_hidden3, size_hidden4)

    def forward(self, input):
        return self.lin4(
            self.relu3(self.lin3(self.relu2(self.lin2(self.relu1(self.lin1(input))))))
        )


def train(model_inp, x_train, y_train, num_epochs=num_epochs):
    datasets = torch.utils.data.TensorDataset(x_train, y_train)
    train_iter = torch.utils.data.DataLoader(datasets, batch_size=10, shuffle=True)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.RMSprop(model_inp.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for inputs, labels in train_iter:
            # forward pass
            outputs = model_inp(inputs)
            # defining loss
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # computing gradients
            loss.backward()
            # accumulating running loss
            running_loss += loss.item()
            # updated weights based on computed gradients
            optimizer.step()
        if epoch % 20 == 0:
            print(
                "Epoch [%d]/[%d] running accumulative loss across all batches: %.3f"
                % (epoch + 1, num_epochs, running_loss)
            )
        running_loss = 0.0


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture(scope="module")
def boston_training_test():
    csv_path = "tests/datasets/boston_housing.csv"
    target_column = "MEDV"
    boston_dataset = pd.read_csv(csv_path)
    y = boston_dataset[target_column]
    x = boston_dataset.drop(target_column, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test


@pytest.fixture()
def sk_function_transformer():
    def simple_preprocessor(data_x):
        return data_x ** 2

    transformer = sk_preprocessing.FunctionTransformer(
        simple_preprocessor, validate=True
    )
    return transformer


@pytest.fixture()
def custom_transformer():
    def simple_preprocessor(data_x):
        transformed_x = data_x + 1.0
        return transformed_x

    return simple_preprocessor


@pytest.fixture()
def add_value_to_column_transformer():
    def drop_column(dataframe_x):
        x_transformed = dataframe_x + dataframe_x
        return x_transformed

    return drop_column


def test_boston_pytorch_no_preprocessing(boston_training_test, model_path):

    x_train, x_test, y_train, _ = boston_training_test

    x_train = torch.Tensor(x_train.values)
    y_train = torch.Tensor(y_train.values)

    x_test = torch.Tensor(x_test.values)

    model = BostonModel()
    model.train()
    train(model, x_train, y_train)

    cbw.save_model(model_path, model)
    loaded_model = cbw.load_model(model_path)

    original_model_predictions = model(x_test).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test).detach().numpy()

    print("== ORIGINAL TYPE: {}".format(type(original_model_predictions)))
    print("== LOADED TYPE: {}".format(type(loaded_model_predictions)))
    print("== ORIGINAL SHAPE: {}".format(original_model_predictions.shape))
    print("== LOADED SHAPE: {}".format(loaded_model_predictions.shape))
    print("== ORIGINAL 5:\n{}".format(original_model_predictions[:5]))
    print("== LOADED 5:\n{}".format(loaded_model_predictions[:5]))

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


@pytest.mark.parametrize(
    "sk_transformer",
    [
        (sk_preprocessing.StandardScaler()),
        (sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50)),
        (sk_preprocessing.KBinsDiscretizer(n_bins=2, encode="ordinal")),
        (sk_preprocessing.RobustScaler()),
        (sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_boston_pytorch_preprocessing(sk_transformer, boston_training_test, model_path):
    x_train, x_test, y_train, _ = boston_training_test

    x_transformed = sk_transformer.fit_transform(x_train)
    x_transformed = torch.Tensor(x_transformed)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = sk_transformer.transform(x_data)
        return torch.Tensor(x_transformed)

    model = BostonModel()
    model.train()
    train(model, x_transformed, y_train)

    cbw.save_model(model_path, model, preprocessing_function)
    loaded_model = cbw.load_model(model_path)

    x_test_transformed = preprocessing_function(x_test)
    original_model_predictions = model(x_test_transformed).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test).detach().numpy()

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_boston_pytorch_preprocessing_with_function_transformer(
    sk_function_transformer, boston_training_test, model_path
):
    x_train, x_test, y_train, _ = boston_training_test

    x_transformed = sk_function_transformer.fit_transform(x_train)
    x_transformed = torch.Tensor(x_transformed)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = sk_function_transformer.transform(x_data)
        return torch.Tensor(x_transformed)

    model = BostonModel()
    model.train()
    train(model, x_transformed, y_train)

    cbw.save_model(model_path, model, preprocessing_function)
    loaded_model = cbw.load_model(model_path)

    x_test_transformed = preprocessing_function(x_test)
    original_model_predictions = model(x_test_transformed).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test).detach().numpy()

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_boston_pytorch_preprocessing_with_custom_transformer(
    custom_transformer, boston_training_test, model_path
):
    x_train, x_test, y_train, _ = boston_training_test

    x_transformed = custom_transformer(x_train)
    x_transformed = torch.Tensor(x_transformed.values)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = custom_transformer(x_data)
        return torch.Tensor(x_transformed.values)

    model = BostonModel()
    model.train()
    train(model, x_transformed, y_train)

    cbw.save_model(model_path, model, preprocessing_function)
    loaded_model = cbw.load_model(model_path)

    x_test_transformed = preprocessing_function(x_test)
    original_model_predictions = model(x_test_transformed).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test).detach().numpy()

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


@pytest.mark.parametrize(
    "preprocessor",
    [
        (sk_preprocessing.StandardScaler()),
        (sk_preprocessing.QuantileTransformer(random_state=0, n_quantiles=50)),
        (sk_preprocessing.KBinsDiscretizer(n_bins=2, encode="ordinal")),
        (sk_preprocessing.RobustScaler()),
        (sk_preprocessing.MaxAbsScaler()),
    ],
)
def test_boston_pytorch__data_cleaning_and_preprocessing(
    preprocessor, add_value_to_column_transformer, boston_training_test, model_path
):
    x_train, x_test, y_train, _ = boston_training_test

    x_cleaned = add_value_to_column_transformer(x_train)
    x_transformed = preprocessor.fit_transform(x_cleaned)

    x_transformed = torch.Tensor(x_transformed)
    y_train = torch.Tensor(y_train.values)

    def preprocessing_function(x_data):
        x_transformed = preprocessor.transform(x_data)
        return torch.Tensor(x_transformed)

    model = BostonModel()
    model.train()
    train(model, x_transformed, y_train)

    cbw.save_model(
        model_path, model, preprocessing_function, add_value_to_column_transformer
    )
    loaded_model = cbw.load_model(model_path)

    x_test_cleaned = add_value_to_column_transformer(x_test)
    x_test_transformed = preprocessing_function(x_test_cleaned)

    original_model_predictions = model(x_test_transformed).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test).detach().numpy()

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)
