import pandas as panda

from models.tf_idf_for_combined_dataset import TFIDFCombined, LogisticRegressionModel, GaussianNaiveBayesModel, \
    RandomForestClassifierModel, MultinomialNaiveBayesModel, BernoulliNaiveBayesModel


def main(combined_dataset):
    # dataset = panda.read_csv("data/t-davidson/labeled_data.csv")
    # train_dataset = panda.read_csv("data/datasets/train.csv")
    # test_dataset = panda.read_csv("data/datasets/test.csv")
    # tf_idf_train = TFIDFCombined(train_dataset)
    # tf_idf_test = TFIDFCombined(test_dataset)

    tf_idf = TFIDFCombined(combined_dataset)
    #tf_idf.visualization()

    LogisticRegressionModel(tf_idf).train()
    GaussianNaiveBayesModel(tf_idf).train()
    RandomForestClassifierModel(tf_idf).train()
    MultinomialNaiveBayesModel(tf_idf).train()
    BernoulliNaiveBayesModel(tf_idf).train()

if __name__ == '__main__':
    dataset_ = panda.read_csv("data/datasets/merged_dataset.csv")
    main(dataset_)
