import string
from statistics import mean
from typing import *

from src.dataUtils import *

analysisOutDir = "../dataset-analysis"


def _count_words(reviews_list) -> Dict[str, int]:
    list_of_lists = [sentence.split(" ") for sentence in reviews_list]
    words = [item for sublist in list_of_lists for item in sublist]
    words_no_punctuation = [_remove_punctuations(word) for word in words]
    words_no_punctuation = list(filter(lambda word: len(word) > 0, words_no_punctuation))
    counts = Counter(words_no_punctuation)
    return counts


def _remove_punctuations(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation)).strip()


def count_words_all() -> None:
    reviews_per_author = read_reviews()
    all_reviews = sum(reviews_per_author.values(), [])
    count_result = _count_words(all_reviews)

    sorted_count_result = _sort_dictionary_desc(count_result)
    save_dict(sorted_count_result, analysisOutDir + "/count-words-all.csv")
    print("Correct saved data - count all words")


def count_words_per_author() -> None:
    reviews_per_author = read_reviews()
    for author in reviews_per_author.keys():
        filepath = analysisOutDir + "/count-words-" + author + ".csv"
        reviews = reviews_per_author[author]
        count_result = _count_words(reviews)
        sorted_count_result = _sort_dictionary_desc(count_result)
        save_dict(sorted_count_result, filepath)
        print("Correct saved data - " + "count words per author" + " - " + author)


def _calc_length(reviews: List[str], filepath: str) -> None:
    reviews_len = [len(review) for review in reviews]
    result = {"avg": mean(reviews_len), "min": min(reviews_len), "max": max(reviews_len)}
    save_dict(result, filepath)
    print("Correct saved data to: " + filepath)


def calc_length_all() -> None:
    reviews_per_author = read_reviews()
    all_reviews = sum(reviews_per_author.values(), [])
    filepath = analysisOutDir + "/calc-length-all.csv"
    _calc_length(all_reviews, filepath)


def calc_length_per_author() -> None:
    reviews_per_author = read_reviews()
    for author in reviews_per_author.keys():
        filepath = analysisOutDir + "/calc-length-" + author + ".csv"
        reviews = reviews_per_author[author]
        _calc_length(reviews, filepath)


def count_labels_all() -> None:
    labels_per_author = read_labels()
    all_labels = sum(labels_per_author.values(), [])
    filepath = analysisOutDir + "/labels-count-all.csv"
    result = Counter(all_labels)
    save_dict(result, filepath)
    print("Correct saved data to: " + filepath)


def count_labels_per_author() -> None:
    labels_per_author = read_labels()
    for author in labels_per_author.keys():
        filepath = analysisOutDir + "/labels-count-" + author + ".csv"
        labels = labels_per_author[author]
        result = Counter(labels)
        save_dict(result, filepath)
        print("Correct saved data to: " + filepath)


def _sort_dictionary_desc(dictionary: Dict[Any, Any]) -> Dict[Any, Any]:
    return dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))
