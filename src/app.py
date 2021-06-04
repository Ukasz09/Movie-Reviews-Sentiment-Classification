from src.dataAnalysis import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

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


def predict_with_bayes(X_train, y_train):
    MNB = MultinomialNB()
    MNB.fit(X_train, y_train)
    return MNB.predict(X_test)


if __name__ == "__main__":
    cv = CountVectorizer(stop_words='english')
    X_raw, y_raw = read_data()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=test_size)
    X_train = cv.fit_transform(X_train_raw)
    X_test = cv.transform(X_test_raw)

    y_predicted = predict_with_bayes(X_train, y_train)

    accuracy_score = metrics.accuracy_score(y_predicted, y_test)
    print(str('{:04.2f}'.format(accuracy_score * 100)) + '%')

    # Dataset analysis
    # count_words_all()
    # count_words_per_author()
    # calc_length_all()
    # calc_length_per_author()
    # count_labels_all()
    # count_labels_per_author()
    # words_qty_per_sentence_all()
    # words_qty_per_sentence_and_author()
