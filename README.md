# ClearBox Wrapper


ClearBox Wrapper is an agnostic wrapper for the most used machine learning frameworks, with the aim of facilitating the transfer of models between different cloud environments and of providing a common interface for generating output predictions.

## Usage

With a few lines of code it is possible to create a wrapper for your model, simply specifying how to perform a prediction and how to carry out input preprocessing operations if necessary.

For example, if you have just trained a model using Sklearn and your input doesn't need preprocessing, just define a class that inherits from SklearnWrapper and specify how to perform the predict method. After that, it will be sufficient to use the Sklearn wrapper _dump_ method to have your model serialized on the disk.


```python
from sklearn.linear_model import LinearRegression

...

lr = lr = LinearRegression()
lr.fit(X_train, y_train)

...

from clearbox_wrapper.SklearnWrapper import SklearnWrapper

class MyModel(SklearnWrapper):    
    def predict(self, X):
        return self.model.predict(X)

MyModel(lr).dump('sklearn_boston.model')
```

At this point you can move the newly created file to any environment you want and simply deserialize it to be able to use it.

```python
from clearbox_wrapper.SklearnWrapper import SklearnWrapper

foo = SklearnWrapper.load('sklearn_boston.model')
foo.predict(data)
```

## Examples

#### Sklearn

* Boston Housing Dataset - [Notebook](https://github.com/ClearBox-AI/clearbox-wrapper/blob/master/examples/sklearn/sklearn_boston_dataset.ipynb)

#### XGBoost

* Pima Indians Diabetes - [Notebook](https://github.com/ClearBox-AI/clearbox-wrapper/blob/master/examples/xgboost/xgboost_diabetes_dataset.ipynb) 

#### PyTorch

* CIFAR-10 - [Notebook](https://github.com/ClearBox-AI/clearbox-wrapper/blob/master/examples/pytorch/pytorch_cifar10_dataset.ipynb)

#### Keras

* Fashion MNIST - [Notebook](https://github.com/ClearBox-AI/clearbox-wrapper/blob/master/examples/keras/keras_fashion_mnist_dataset.ipynb)

## License

[Apache License 2.0](https://github.com/ClearBox-AI/clearbox-wrapper/blob/master/LICENSE)
