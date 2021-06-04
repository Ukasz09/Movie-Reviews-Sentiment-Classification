from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.dataAnalysis import *
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics, svm


def make_list(raw_list, indexes):
    data = []
    for index in indexes:
        data.append(raw_list[index])
    return data


def predict_with_bayes(X_train, X_test, y_train):
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    return mnb.predict(X_test)


def predict_with_svm(X_train, X_test, y_train):
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(X_train, y_train)
    return classifier_linear.predict(X_test)


def predict_with_k_cross_validation(X_raw, y_raw, predict_func, vectorizer, k_fold_groups=10):
    k_fold = KFold(k_fold_groups, shuffle=True)
    accuracy_sum = 0
    i = 0
    for train_indexes, test_indexes in k_fold.split(X_raw):
        X_train_raw, X_test_raw = make_list(X_raw, train_indexes), make_list(X_raw,
                                                                             test_indexes)
        y_train, y_test = make_list(y_raw, train_indexes), make_list(y_raw, test_indexes)
        X_train, X_test = vectorizer.fit_transform(X_train_raw), vectorizer.transform(X_test_raw)
        y_predicted = predict_func(X_train, X_test, y_train)
        accuracy_score = metrics.accuracy_score(y_predicted, y_test)
        accuracy_sum += accuracy_score
        i += 1
        print('Fold=', i, str('Mean accuracy = {:04.2f}'.format(accuracy_score * 100)), '%')
    mean_accuracy = accuracy_sum / k_fold_groups
    print(str('Mean accuracy = {:04.2f}'.format(mean_accuracy * 100)) + '%')


if __name__ == "__main__":
    X_raw, y_raw = read_data()
    tv = TfidfVectorizer(stop_words='english')
    cv = CountVectorizer(stop_words='english')

    print("\nNaive Bayes, CountVectorizer\n")
    predict_with_k_cross_validation(X_raw, y_raw, predict_with_bayes, cv)

    print("\nNaive Bayes, TfidfVectorizer\n")
    predict_with_k_cross_validation(X_raw, y_raw, predict_with_bayes, tv)

    print("\nSVM, CountVectorizer\n")
    predict_with_k_cross_validation(X_raw, y_raw, predict_with_svm, cv)

    print("\nSVM, TfidfVectorizer\n")
    predict_with_k_cross_validation(X_raw, y_raw, predict_with_svm, tv)
