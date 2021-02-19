"""
This module imports contents from CloudPickle in a way that is compatible with the
``pickle_module`` parameter of PyTorch's model persistence function: ``torch.save``
(see https://github.com/pytorch/pytorch/blob/692898fe379c9092f5e380797c32305145cd06e1/torch/
serialization.py#L192). It is included as a distinct module from :mod:`mlflow.pytorch` to avoid
polluting the namespace with wildcard imports.

Calling ``torch.save(..., pickle_module=mlflow.pytorch.pickle_module)`` will persist PyTorch
definitions using CloudPickle, leveraging improved pickling functionality such as the ability
to capture class definitions in the "__main__" scope.

TODO: Remove this module or make it an alias of CloudPickle when CloudPickle and PyTorch have
compatible pickling APIs.
"""

from cloudpickle import *  # noqa


from cloudpickle import CloudPickler as Pickler  # noqa


from pickle import Unpickler  # noqa
