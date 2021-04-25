from abc import abstractmethod, ABC

from pandas import Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from helpers.preprocessing import AbstractProcessor, stemming, remove_stopwords, get_tokens, remove_mentions, remove_additional_spaces, \
    keep_chars_only


class TFIDF(AbstractProcessor):
    dataset: Series
    additional_stopwords: []
    vector: TfidfVectorizer
    tfidf: None

    def __init__(self, dataset):
        self.dataset = dataset
        self.additional_stopwords = ['rt']
        self.preprocess()
        self.vector = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=10000)
        self.tfidf = self.vector.fit_transform(dataset['processed_tweets'])

    def preprocess(self):
        tweet = self.dataset.tweet
        tweet = tweet.str.lower()

        tweet = remove_mentions(tweet)
        tweet = keep_chars_only(tweet)
        tweet = remove_additional_spaces(tweet)

        # get tokens, remove stopwords and stem
        tweet_tokens = stemming(remove_stopwords(get_tokens(tweet), self.additional_stopwords))

        for i in range(len(tweet_tokens)):
            tweet_tokens[i] = ' '.join(tweet_tokens[i])
            tweets = tweet_tokens

        self.dataset['processed_tweets'] = tweets

    def visualization(self):
        all_words = ' '.join(
            [text for text in self.dataset['processed_tweets'][self.dataset['class'] == 0][self.dataset['hate_speech'] == 3]]
        )
        wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()


class Model(ABC):
    tfidf: TFIDF
    train_tfidf: None
    test_tfidf: None
    train_labels: None
    test_labels: None

    def __init__(self, tfidf):
        self.tfidf = tfidf

    @abstractmethod
    def train(self):
        classes = self.tfidf.dataset['class'].astype(int)
        self.train_tfidf, self.test_tfidf, self.train_labels, self.test_labels = train_test_split(
            self.tfidf.tfidf.toarray(), classes, random_state=42, test_size=0.3
        )

    def score(self, class_name):
        y_preds = self.model.predict(self.test_tfidf)
        acc = accuracy_score(self.test_labels, y_preds)
        print(f"{class_name}, Accuracy Score:", acc)


class LogisticRegressionModel(Model):
    def __init__(self, tfidf: TFIDF):
        super().__init__(tfidf)

    def train(self):
        super().train()
        self.model = LogisticRegression(multi_class='auto', solver='newton-cg',)
        self.model.fit(self.train_tfidf, self.train_labels)
        super().score(__class__.__name__ )


class MultinomialNaiveBayesModel(Model):
    def __init__(self, tfidf: TFIDF):
        super().__init__(tfidf)

    def train(self):
        super().train()
        self.model = MultinomialNB(alpha=0.75)
        self.model.fit(self.train_tfidf, self.train_labels)

        super().score(__class__.__name__ )


class GaussianNaiveBayesModel(Model):
    def __init__(self, tfidf: TFIDF):
        super().__init__(tfidf)

    def train(self):
        super().train()
        self.model = GaussianNB()
        self.model.fit(self.train_tfidf, self.train_labels)

        super().score(__class__.__name__ )


class RandomForestClassifierModel(Model):
    def __init__(self, tfidf: TFIDF):
        super().__init__(tfidf)

    def train(self):
        super().train()
        self.model = RandomForestClassifier()
        self.model.fit(self.train_tfidf, self.train_labels)

        super().score(__class__.__name__ )


class BernoulliNaiveBayesModel(Model):
    def __init__(self, tfidf: TFIDF):
        super().__init__(tfidf)

    def train(self):
        super().train()
        self.model = BernoulliNB()
        self.model.fit(self.train_tfidf, self.train_labels)

        super().score(__class__.__name__ )
