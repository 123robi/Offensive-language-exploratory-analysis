import pandas as panda
from models.custom_classifier import main
import sys

if __name__ == '__main__':
    # provide argument `subtype` if you want to change to multilabel
    argument = 'hatespeech'
    db = 'slovene'

    if len(sys.argv) == 2:
        argument = sys.argv[1]
    if len(sys.argv) == 3:
        db = sys.argv[2]

    if db == 'slovene':
        train_dataset = panda.read_csv("../data/datasets/slovene_dataset/train.csv")
        test_dataset = panda.read_csv("../data/datasets/slovene_dataset/test.csv")
    else:
        train_dataset = panda.read_csv("data/datasets/english_dataset/train.csv")
        test_dataset = panda.read_csv("data/datasets/english_dataset/test.csv")

    dataset = test_dataset.append(train_dataset, ignore_index=True)

    main(dataset, argument)
