import os
from typing import *

subjectFilesPrefix = "subj."
dataFilesPath = "../data"


def get_list_of_files(dir_name) -> List[str]:
    list_of_file = os.listdir(dir_name)
    all_files = list()
    for entry in list_of_file:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            all_files.append(full_path)

    return all_files


def is_subject_file(path: str) -> bool:
    return path.split("/")[-1].startswith(subjectFilesPrefix)


def get_subject_files() -> Iterator:
    files_paths = get_list_of_files(dataFilesPath)
    return filter(lambda path: is_subject_file(path), files_paths)


if __name__ == "__main__":
    test2 = get_subject_files()
    for t in test2:
        print(t)
