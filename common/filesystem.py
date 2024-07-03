import os
import os.path as osp
from glob import glob


def may_make_dir(path):
    """
    make a new dir if it not exists
    :param path: a dir, or result of `osp.dirname(osp.abspath(file_path))
    :return:
    """
    assert path not in [None, '']

    if not osp.exists(path):
        os.makedirs(path)

    return path


def scan_dir(path, recursive=False, exts=('jpg', 'jpeg', 'png')):
    files = []

    for ext in exts:
        if recursive:
            patter = osp.join(path, '**', f'*.{ext}')
        else:
            patter = osp.join(path, f'*.{ext}')
        files.extend(glob(patter, recursive=recursive))

    return len(files), files
