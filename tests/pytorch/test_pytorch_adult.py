import os

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import sklearn.preprocessing as sk_preprocessing
import torch
import torch.nn as nn
import torch.optim as optim

import clearbox_wrapper as cbw

num_epochs = 20
learning_rate = 0.001
size_hidden1 = 25
size_hidden2 = 12
size_hidden3 = 6
size_hidden4 = 2


class AdultModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.lin1 = nn.Linear(input_shape, size_hidden1)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(size_hidden1, size_hidden2)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(size_hidden2, size_hidden3)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(size_hidden3, size_hidden4)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input):
        return self.softmax(
            self.lin4(
                self.relu3(
                    self.lin3(self.relu2(self.lin2(self.relu1(self.lin1(input)))))
                )
            )
        )


def train(model, x_train, y_train, num_epochs=num_epochs):
    datasets = torch.utils.data.TensorDataset(x_train, y_train)
    train_iter = torch.utils.data.DataLoader(datasets, batch_size=64, shuffle=True)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for inputs, labels in train_iter:
            # forward pass
            outputs = model(inputs)
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
    y_encoder = sk_preprocessing.OneHotEncoder()
    return x_encoder, y_encoder


def test_adult_pytorch_preprocessing(adult_training, adult_test, model_path):
    x_training, y_training = adult_training
    x_test, _ = adult_test
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training)

    x_train_transformed = x_preprocessor.fit_transform(x_training)
    x_train_transformed = torch.Tensor(x_train_transformed.todense())

    y_train_transformed = y_encoder.fit_transform(y_training.values.reshape(-1, 1))
    y_train_transformed = torch.Tensor(y_train_transformed.todense())

    def preprocessing_function(x_data):
        x_transformed = x_preprocessor.transform(x_data)
        return x_transformed.todense()

    model = AdultModel(x_train_transformed.shape[1])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train(model, x_train_transformed, y_train_transformed)

    cbw.save_model(model_path, model, preprocessing=preprocessing_function, zip=False)
    loaded_model = cbw.load_model(model_path)

    x_test_transformed = preprocessing_function(x_test)
    x_test_transformed = torch.Tensor(x_test_transformed)
    original_model_predictions = model(x_test_transformed).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test)

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)


def test_adult_pytorch_preprocessing_and_data_preparation(
    adult_training, adult_test, data_preparation, model_path
):
    x_training, y_training = adult_training
    x_test, _ = adult_test

    x_training_prepared = data_preparation(x_training)
    x_preprocessor, y_encoder = x_and_y_preprocessing(x_training_prepared)

    x_train_transformed = x_preprocessor.fit_transform(x_training_prepared)
    x_train_transformed = torch.Tensor(x_train_transformed.todense())

    y_train_transformed = y_encoder.fit_transform(y_training.values.reshape(-1, 1))
    y_train_transformed = torch.Tensor(y_train_transformed.todense())

    def preprocessing_function(x_data):
        x_transformed = x_preprocessor.transform(x_data)
        return x_transformed.todense()

    model = AdultModel(x_train_transformed.shape[1])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train(model, x_train_transformed, y_train_transformed)

    cbw.save_model(
        model_path,
        model,
        preprocessing=preprocessing_function,
        data_preparation=data_preparation,
        zip=False,
    )
    loaded_model = cbw.load_model(model_path)

    x_test_prepared = data_preparation(x_test)
    x_test_transformed = preprocessing_function(x_test_prepared)
    x_test_transformed = torch.Tensor(x_test_transformed)

    original_model_predictions = model(x_test_transformed).detach().numpy()
    loaded_model_predictions = loaded_model.predict(x_test)

    np.testing.assert_array_equal(original_model_predictions, loaded_model_predictions)
