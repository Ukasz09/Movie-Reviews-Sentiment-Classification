from sklearn.feature_extraction.text import CountVectorizer
from typing import *


def count_subjects(reviews_list) -> Dict[str, int]:
    vectorizer = CountVectorizer()
    vectorizer.fit(reviews_list)
    vocabulary = vectorizer.vocabulary_
    return vocabulary
