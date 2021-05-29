from sklearn.feature_extraction.text import CountVectorizer


def count_subjects(reviews_list):
    vectorizer = CountVectorizer()
    vectorizer.fit(reviews_list)
    vocabulary = vectorizer.vocabulary_
    return vocabulary
