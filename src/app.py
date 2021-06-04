from src.dataAnalysis import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

negative_sentiment = 0
neutral_sentiment = 1
positive_sentiment = 2


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


def predict_with_bayes(X_train, X_test, y_train):
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    return mnb.predict(X_test)


def split_normal(test_size=0.2):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=test_size)
    return X_train_raw, X_test_raw, y_train, y_test


def get_from_raw(raw, indexes):
    data = []
    for index in indexes:
        data.append(raw[index])
    return data


if __name__ == "__main__":
    cv = CountVectorizer(stop_words='english')
    X_raw, y_raw = read_data()

    k_fold_groups = 10
    k_fold = KFold(k_fold_groups, shuffle=True)
    accuracy_sum = 0
    i = 0
    for train_indexes, test_indexes in k_fold.split(X_raw):
        X_train_raw, X_test_raw = get_from_raw(X_raw, train_indexes), get_from_raw(X_raw, test_indexes)
        y_train, y_test = get_from_raw(y_raw, train_indexes), get_from_raw(y_raw, test_indexes)
        X_train, X_test = cv.fit_transform(X_train_raw), cv.transform(X_test_raw)
        y_predicted = predict_with_bayes(X_train, X_test, y_train)
        accuracy_score = metrics.accuracy_score(y_predicted, y_test)
        accuracy_sum += accuracy_score
        i += 1
        print('Fold=', i, str('Mean accuracy = {:04.2f}'.format(accuracy_score * 100)), '%')
    mean_accuracy = accuracy_sum / k_fold_groups
    print('\n', str('Mean accuracy = {:04.2f}'.format(mean_accuracy * 100)) + '%')

    # Dataset analysis
    # count_words_all()
    # count_words_per_author()
    # calc_length_all()
    # calc_length_per_author()
    # count_labels_all()
    # count_labels_per_author()
    # words_qty_per_sentence_all()
    # words_qty_per_sentence_and_author()
