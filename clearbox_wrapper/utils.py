import os
import shutil
import zipfile


def zip_directory(directory_path: str) -> None:
    """Given a directory path, zip the directory.

    Parameters
    ----------
    directory_path : str
        Directory path
    """
    zip_object = zipfile.ZipFile(directory_path + ".zip", "w", zipfile.ZIP_DEFLATED)
    root_len = len(directory_path) + 1
    for base, _dirs, files in os.walk(directory_path):
        for file in files:
            fn = os.path.join(base, file)
            zip_object.write(fn, fn[root_len:])
    shutil.rmtree(directory_path)
