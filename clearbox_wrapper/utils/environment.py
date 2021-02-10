from sys import version_info
from typing import Dict, List, Optional, Union

import yaml

CONDA_HEADER = """\
name: conda-env
channels:
  - defaults
  - conda-forge
"""

PYTHON_VERSION = "{major}.{minor}.{micro}".format(
    major=version_info.major, minor=version_info.minor, micro=version_info.micro
)


def get_major_minor_py_version(py_version):
    return ".".join(py_version.split(".")[:2])


def _get_default_conda_env(
    path: str = None,
    additional_conda_deps: Optional[List] = None,
    additional_pip_deps: Optional[List] = None,
    additional_conda_channels: Optional[List] = None,
) -> Union[Dict, None]:
    """Generate and, optionally, save to file the default Conda environment for models.

    Parameters
    ----------
    path : str, optional
        File path. If not None, the Conda env will be saved to file, by default None
    additional_conda_deps : Optional[List], optional
        List of additional conda dependencies, by default None
    additional_pip_deps : Optional[List], optional
        List of additional Pypi dependencies, by default None
    additional_conda_channels : Optional[List], optional
        List of additional conda channels, by default None

    Returns
    -------
    Union[Dict, None]
        None if path is not None, else the Conda environment generated as a dictionary.
    """
    pip_deps = additional_pip_deps if additional_pip_deps else []
    conda_deps = (additional_conda_deps if additional_conda_deps else []) + (
        ["pip"] if pip_deps else []
    )

    env = yaml.safe_load(CONDA_HEADER)
    env["dependencies"] = ["python={}".format(PYTHON_VERSION)]
    if conda_deps is not None:
        env["dependencies"] += conda_deps
    env["dependencies"].append({"pip": pip_deps})
    if additional_conda_channels is not None:
        env["channels"] += additional_conda_channels

    if path is not None:
        with open(path, "w") as out:
            yaml.safe_dump(env, stream=out, default_flow_style=False)
        return None
    else:
        return env
