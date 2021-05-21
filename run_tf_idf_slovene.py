import pandas as panda

from models.tf_idf_for_slovene import TFIDFSlovene, LogisticRegressionModel, GaussianNaiveBayesModel, \
    RandomForestClassifierModel, MultinomialNaiveBayesModel, BernoulliNaiveBayesModel


def main():
    # dataset = panda.read_csv("data/t-davidson/labeled_data.csv")
    # train_dataset = panda.read_csv("data/datasets/train.csv")
    # test_dataset = panda.read_csv("data/datasets/test.csv")
    # tf_idf_train = TFIDFCombined(train_dataset)
    # tf_idf_test = TFIDFCombined(test_dataset)


    train_data = panda.read_csv("data/datasets/slovene_dataset/train.csv")
    test_data = panda.read_csv("data/datasets/slovene_dataset/test.csv")

    frames = [train_data, test_data]
    combined_data = panda.concat(frames)

    tf_idf = TFIDFSlovene(combined_data)
    tf_idf.visualization()

    LogisticRegressionModel(tf_idf).train()
    GaussianNaiveBayesModel(tf_idf).train()
    RandomForestClassifierModel(tf_idf).train()
    MultinomialNaiveBayesModel(tf_idf).train()
    BernoulliNaiveBayesModel(tf_idf).train()

if __name__ == '__main__':
    main()
