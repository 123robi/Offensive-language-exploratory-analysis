from abc import abstractmethod, ABC

from pandas import Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from helpers.preprocessing import AbstractProcessor, stemming, remove_stopwords, get_tokens, remove_mentions, \
    remove_additional_spaces, \
    keep_chars_only, remove_stopwords_slovene, remove_mentions_slovene, keep_chars_only_slovene, remove_urls

import numpy as np

class TFIDFSlovene(AbstractProcessor):
    dataset: Series
    additional_stopwords: []
    vector: TfidfVectorizer
    tfidf: None

    def __init__(self, dataset):
        self.dataset = dataset
        self.additional_stopwords = ['rt']
        self.preprocess()
        self.vector = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=10000)
        self.tfidf = self.vector.fit_transform(dataset['processed_text'].apply(lambda x: np.str_(x)))

    def preprocess(self):
        text = self.dataset.text
        text = text.str.lower()

        text = remove_mentions_slovene(text)
        text = remove_urls(text)
        text = keep_chars_only_slovene(text)
        text = remove_additional_spaces(text)
        text.dropna(inplace=True)
        # get tokens, remove stopwords
        text_tokens = remove_stopwords_slovene(get_tokens(text), self.additional_stopwords)

        for i in range(len(text_tokens)):
            try:
                text_tokens[i] = ' '.join(text_tokens[i])
            except:
                print("we get key errors, idk why, at", i)

        text = text_tokens
        self.dataset['processed_text'] = text

    def visualization(self):

        all_words = ' '.join(
            [str(text) for text in self.dataset['processed_text'][self.dataset['hatespeech'] == 1]]
        )
        wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()


class ModelSlovene(ABC):
    tfidf: TFIDFSlovene
    train_tfidf: None
    test_tfidf: None
    train_labels: None
    test_labels: None

    def __init__(self, tfidf, is_binary):
        self.tfidf = tfidf
        self.is_binary = is_binary

    @abstractmethod
    def train(self):
        if self.is_binary:
            classes = self.tfidf.dataset['hatespeech'].astype(int)
        else:
            classes = self.tfidf.dataset['subtype'].astype(int)

        self.train_tfidf, self.test_tfidf, self.train_labels, self.test_labels = train_test_split(
            self.tfidf.tfidf.toarray(), classes, random_state=42, test_size=0.3
        )

    def score(self, class_name):
        y_preds = self.model.predict(self.test_tfidf)
        acc = accuracy_score(self.test_labels, y_preds)
       # print(f"{class_name}, Accuracy Score:", acc)

        precision = precision_score(self.test_labels, y_preds, average="weighted")
        recall = recall_score(self.test_labels, y_preds, average="weighted")
        f_score = f1_score(self.test_labels, y_preds, average="weighted")

        confusion_mat = confusion_matrix(self.test_labels, y_preds)
        print(f"Model name: {class_name}")
        print("\t Accuracy Score:", acc)
        print("\t Precision Score:", precision)
        print("\t Recall Score:", recall)
        print("\t F-Score:", f_score)
        print("\t Confusion Matrix:", confusion_mat)



class LogisticRegressionModel(ModelSlovene):
    def __init__(self, tfidf: TFIDFSlovene, is_binary):
        super().__init__(tfidf, is_binary)

    def train(self):
        super().train()
        self.model = LogisticRegression(multi_class='auto', solver='newton-cg',)
        self.model.fit(self.train_tfidf, self.train_labels)
        super().score(__class__.__name__ )


class MultinomialNaiveBayesModel(ModelSlovene):
    def __init__(self, tfidf: TFIDFSlovene, is_binary):
        super().__init__(tfidf, is_binary)

    def train(self):
        super().train()
        self.model = MultinomialNB(alpha=0.75)
        self.model.fit(self.train_tfidf, self.train_labels)

        super().score(__class__.__name__ )


class GaussianNaiveBayesModel(ModelSlovene):
    def __init__(self, tfidf: TFIDFSlovene, is_binary):
        super().__init__(tfidf, is_binary)

    def train(self):
        super().train()
        self.model = GaussianNB()
        self.model.fit(self.train_tfidf, self.train_labels)

        super().score(__class__.__name__ )


class RandomForestClassifierModel(ModelSlovene):
    def __init__(self, tfidf: TFIDFSlovene, is_binary):
        super().__init__(tfidf, is_binary)

    def train(self):
        super().train()
        self.model = RandomForestClassifier()
        self.model.fit(self.train_tfidf, self.train_labels)

        super().score(__class__.__name__ )


class BernoulliNaiveBayesModel(ModelSlovene):
    def __init__(self, tfidf: TFIDFSlovene, is_binary):
        super().__init__(tfidf, is_binary)

    def train(self):
        super().train()
        self.model = BernoulliNB()
        self.model.fit(self.train_tfidf, self.train_labels)

        super().score(__class__.__name__ )
