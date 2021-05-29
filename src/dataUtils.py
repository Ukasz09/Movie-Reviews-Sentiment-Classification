import os
from typing import *

subject_files_prefix = "subj."
label_files_prefix = "label.3class."
data_path = "../data"


def _get_list_of_files(dir_name: str) -> List[str]:
    list_of_file = os.listdir(dir_name)
    all_files = list()
    for entry in list_of_file:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            all_files = all_files + _get_list_of_files(full_path)
        else:
            all_files.append(full_path)
    return all_files


def _is_review_file(path: str) -> bool:
    return path.split("/")[-1].startswith(subject_files_prefix)


def _is_label_file(path: str) -> bool:
    return path.split("/")[-1].startswith(label_files_prefix)


def _get_review_files() -> Iterator:
    files_paths = _get_list_of_files(data_path)
    return filter(lambda path: _is_review_file(path), files_paths)


def _get_label_files() -> Iterator:
    files_paths = _get_list_of_files(data_path)
    return filter(lambda path: _is_label_file(path), files_paths)


def _get_author(filepath: str) -> str:
    return filepath.split("/")[-1].split(".")[-1]


def read_reviews() -> Dict[str, List[str]]:
    """
    @:return Dictionary (author, list of reviews)
    """
    reviews_dict = {}
    for path in _get_review_files():
        author = _get_author(path)
        with open(path) as f:
            reviews = f.readlines()
            reviews_dict[author] = reviews
    return reviews_dict


def read_labels() -> Dict[str, List[int]]:
    """
    @:return Dictionary (author, list of reviews)
    """
    labels_dict = {}
    for path in _get_label_files():
        author = _get_author(path)
        with open(path) as f:
            labels = [int(label_str) for label_str in f.readlines()]
            labels_dict[author] = labels
    return labels_dict
