from typing import *
from src.dataAnalysis import *

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


def get_vectorizer(reviews_list):
    vectorizer = CountVectorizer()
    vectorizer.fit(reviews_list)
    return vectorizer


if __name__ == "__main__":
    # CountVectorizer
    reviews_per_author = read_reviews()
    all_reviews = sum(reviews_per_author.values(), [])
    data = get_vectorizer(all_reviews)
    vocabulary = data.vocabulary_

    # Dataset analysis
    count_words_all()
    count_words_per_author()
    calc_length_all()
    calc_length_per_author()
    count_labels_all()
    count_labels_per_author()
    words_qty_per_sentence_all()
    words_qty_per_sentence_and_author()
