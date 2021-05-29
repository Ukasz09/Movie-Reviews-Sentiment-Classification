import os
from typing import *
from itertools import islice
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

subject_files_prefix = "subj."
label_files_prefix = "label.3class."
data_path = "../data"

negative_sentiment = 0
neutral_sentiment = 1
positive_sentiment = 2

train_size = 0.8
test_size = 0.2


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
    return path.split("/")[-1].startswith(subject_files_prefix)


def is_label_file(path):
    return path.split("/")[-1].startswith(label_files_prefix)


def get_subject_files() -> Iterator:
    files_paths = get_list_of_files(data_path)
    return filter(lambda path: is_subject_file(path), files_paths)


def get_label_files():
    files_paths = get_list_of_files(data_path)
    return filter(lambda path: is_label_file(path), files_paths)


def get_author(filepath: str) -> str:
    return filepath.split("/")[-1].split(".")[-1]


def read_reviews(file_paths: List[str]) -> Dict[str, List[str]]:
    """
    @:return Dictionary (author, list of reviews)
    """
    reviews_dict = {}
    for path in file_paths:
        author = get_author(path)
        with open(path) as f:
            reviews = f.readlines()
            reviews_dict[author] = reviews
    return reviews_dict


def read_labels(file_paths: List[str]) -> Dict[str, List[int]]:
    """
    @:return Dictionary (author, list of reviews)
    """
    labels_dict = {}
    for path in file_paths:
        author = get_author(path)
        with open(path) as f:
            labels = [int(label_str) for label_str in f.readlines()]
            labels_dict[author] = labels
    return labels_dict


def count_subjects(reviews_list):
    vectorizer = CountVectorizer()
    vectorizer.fit(reviews_list)
    vocabulary = vectorizer.vocabulary_
    return vocabulary


def get_data():
    """
    :return: Dictionary (sentence, sentiment)
    """
    subject_files = get_subject_files()
    label_files = get_label_files()
    reviews_per_author = read_reviews(subject_files)
    labels_per_author = read_labels(label_files)
    data = {}
    for author in reviews_per_author.keys():
        reviews = reviews_per_author[author]
        labels = labels_per_author[author]
        for r, l in zip(reviews, labels):
            data[r] = l
    return data


def split_data(data: Dict[str, int]):
    data_tuples = [(v, k) for k, v in data.items()]
    train, test = train_test_split(data_tuples, test_size=test_size, train_size=train_size)
    return train, test


if __name__ == "__main__":
    # subject_files = get_subject_files()
    # reviews_dict = read_reviews(subject_files)
    # all_reviews = sum(reviews_dict.values(), [])
    # print(count_subjects(all_reviews))
    data = get_data()
    train_data, test_data = split_data(data)
