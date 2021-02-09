import os
import shutil
import tempfile


def _copy_file_or_tree(src, dst, dst_dir=None):
    """
    :return: The path to the copied artifacts, relative to `dst`
    """
    dst_subpath = os.path.basename(os.path.abspath(src))
    if dst_dir is not None:
        dst_subpath = os.path.join(dst_dir, dst_subpath)
    dst_path = os.path.join(dst, dst_subpath)
    if os.path.isfile(src):
        dst_dirpath = os.path.dirname(dst_path)
        if not os.path.exists(dst_dirpath):
            os.makedirs(dst_dirpath)
        shutil.copy(src=src, dst=dst_path)
    else:
        shutil.copytree(src=src, dst=dst_path)
    return dst_subpath


class TempDir(object):
    def __init__(self, chdr=False, remove_on_exit=True):
        self._dir = None
        self._path = None
        self._chdr = chdr
        self._remove = remove_on_exit

    def __enter__(self):
        self._path = os.path.abspath(tempfile.mkdtemp())
        assert os.path.exists(self._path)
        if self._chdr:
            self._dir = os.path.abspath(os.getcwd())
            os.chdir(self._path)
        return self

    def __exit__(self, tp, val, traceback):
        if self._chdr and self._dir:
            os.chdir(self._dir)
            self._dir = None
        if self._remove and os.path.exists(self._path):
            shutil.rmtree(self._path)

        assert not self._remove or not os.path.exists(self._path)
        assert os.path.exists(os.getcwd())

    def path(self, *path):
        return (
            os.path.join("./", *path) if self._chdr else os.path.join(self._path, *path)
        )
