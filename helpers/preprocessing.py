import re
from abc import ABC, abstractmethod

import nltk


class AbstractProcessor(ABC):
    @abstractmethod
    def preprocess(self):
        pass


stemmer = nltk.SnowballStemmer("english")


def get_tokens(data):
    return data.apply(nltk.word_tokenize)


def remove_stopwords(data, additional_stopwords):
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords.extend(additional_stopwords)
    return data.apply(lambda tokens: [token for token in tokens if token not in stopwords])


def stemming(data):
    return data.apply(lambda tokens: [stemmer.stem(token) for token in tokens])


def remove_mentions(data):
    regex = re.compile(r"/@[A-Za-z0-9_-]*/g")
    return data.str.replace(regex, "")


def remove_additional_spaces(data):
    regex = re.compile("r\s+")
    return data.str.replace(regex, " ")


def keep_chars_only(data):
    regex = re.compile("[^a-zA-Z]")
    return data.str.replace(regex, " ")


def single_string_remove_mention(data):
    regex = re.compile(r"/@[A-Za-z0-9_-]*/g")
    return regex.sub("", data)


def single_keep_chars_only(data):
    regex = re.compile("[^a-zA-Z]")
    return regex.sub(" ", data)


def single_remove_spaces(data):
    regex = re.compile("r\s+")
    return regex.sub(" ", data)

