import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from helpers.preprocessing import single_string_remove_mention, single_keep_chars_only, single_remove_spaces


def get_features(tweet):
    processed_tweet = single_remove_spaces(single_string_remove_mention(single_keep_chars_only(tweet)))
    num_chars = sum(len(w) for w in processed_tweet)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(processed_tweet.split())
    retweet = 0
    if "rt" in tweet:
        retweet = 1

    return [num_chars, num_chars_total, num_terms, num_words, retweet]


class AbstractCustomClassifier():
    classes: []
    features: []
    model: None
    train_vec: []
    test_vec: []
    train_labels: []
    test_labels: []

    def __init__(self, features, classes):
        self.features = features
        self.classes = classes
        self.train_vec, self.test_vec, self.train_labels, self.test_labels = train_test_split(
            self.features, self.classes, random_state=42, test_size=0.3
        )

    def score(self):
        y_preds = self.model.predict(self.test_vec)
        acc = accuracy_score(self.test_labels, y_preds)

        precision = precision_score(self.test_labels, y_preds, average="weighted")
        recall = recall_score(self.test_labels, y_preds, average="weighted")
        f_score = f1_score(self.test_labels, y_preds, average="weighted")

        print(self.model.__class__.__name__ )
        print("\t Accuracy Score:", acc)
        print("\t Precision Score:", precision)
        print("\t Recall Score:", recall)
        print("\t F-Score:", f_score)


class CustomClassifier(AbstractCustomClassifier):

    def __init__(self, features, classes, model):
        self.model = model
        super().__init__(features, classes)

    def train(self):
        self.model.fit(self.train_vec, self.train_labels)
        super().score()


def main(dataset):
    features = []
    classes = []
    for i, tweet in enumerate(dataset.tweet):
        classes.append(dataset['class'][i])
        features.append(get_features(tweet))

    X, y = np.array(features), np.array(classes)

    CustomClassifier(X, y, MultinomialNB(alpha=.01)).train()
    CustomClassifier(X, y, GaussianNB()).train()
    CustomClassifier(X, y, RandomForestClassifier()).train()
    CustomClassifier(X, y, BernoulliNB()).train()
    CustomClassifier(X, y, LogisticRegression(multi_class='auto', solver='newton-cg',)).train()
