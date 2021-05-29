from typing import *
from sklearn.model_selection import train_test_split
from src.dataUtils import *

negative_sentiment = 0
neutral_sentiment = 1
positive_sentiment = 2

train_size = 0.8
test_size = 0.2


def get_data() -> Dict[str, int]:
    """
    :return: Dictionary (sentence, sentiment)
    """
    reviews_per_author = read_reviews()
    labels_per_author = read_labels()
    data = {}
    for author in reviews_per_author.keys():
        reviews = reviews_per_author[author]
        labels = labels_per_author[author]
        for r, l in zip(reviews, labels):
            data[r] = l
    return data


def split_data(data: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
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
