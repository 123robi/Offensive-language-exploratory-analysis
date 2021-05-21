import pandas as panda
from models.tf_idf import TFIDF, LogisticRegressionModel, GaussianNaiveBayesModel, RandomForestClassifierModel, \
    MultinomialNaiveBayesModel, BernoulliNaiveBayesModel


def main():
    dataset = panda.read_csv("data/t-davidson/labeled_data.csv")
    tf_idf = TFIDF(dataset)
    #tf_idf.preprocess()
    #tf_idf.visualization()

    LogisticRegressionModel(tf_idf).train()
    GaussianNaiveBayesModel(tf_idf).train()
    RandomForestClassifierModel(tf_idf).train()
    MultinomialNaiveBayesModel(tf_idf).train()
    BernoulliNaiveBayesModel(tf_idf).train()


if __name__ == '__main__':
    main()
