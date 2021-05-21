import pandas as panda
from models.custom_classifier import main

if __name__ == '__main__':
    dataset = panda.read_csv("data/t-davidson/labeled_data.csv")
    main(dataset)
