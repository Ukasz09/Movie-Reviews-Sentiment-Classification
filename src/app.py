from typing import *
from src.dataAnalysis import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

negative_sentiment = 0
neutral_sentiment = 1
positive_sentiment = 2

test_size = 0.2


def read_data() -> Tuple[List[str], List[int]]:
    """
    :return: Dictionary (sentence, sentiment)
    """
    reviews_per_author = read_reviews()
    labels_per_author = read_labels()
    all_reviews = []
    all_sentiments = []
    for author in reviews_per_author.keys():
        reviews = reviews_per_author[author]
        labels = labels_per_author[author]
        all_reviews += reviews
        all_sentiments += labels
    return all_reviews, all_sentiments


if __name__ == "__main__":
    X, Y = read_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    cv = CountVectorizer(stop_words='english')
    cvFitTrans = cv.fit_transform(X_train)
    cvTestTrans = cv.transform(X_test)

    # Dataset analysis
    # count_words_all()
    # count_words_per_author()
    # calc_length_all()
    # calc_length_per_author()
    # count_labels_all()
    # count_labels_per_author()
    # words_qty_per_sentence_all()
    # words_qty_per_sentence_and_author()
