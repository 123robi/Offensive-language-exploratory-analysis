import pandas as pd
import numpy as np
import keras.backend as K
from helpers.utils_elmo import *
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def run_elmo(dataset, weights):
    test_set = pd.read_csv("data/transformed_datasets/test.csv", encoding='utf-8')
    X = test_set['text']
    y = test_set['hatespeech']

    model_elmo = build_model()

    predictions = []
    true_labels = []
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        model_elmo.load_weights('models/ELMo/model_elmo_weights-B.h5')
        import time
        t = time.time()
        for i in range(len(X)):
            predicts = model_elmo.predict(np.array([str(X[i]), str(X[i])], dtype=object)[:, np.newaxis])
            if predicts[0] > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
            true_labels.append(y[i])

        print(predictions)
        print(true_labels)
        acc = np.sum(np.array(predictions) == np.array(true_labels)) / len(predictions)
        print(acc)
        print("Accuracy: ", accuracy_score(true_labels, predictions))
        print("Precision: ", precision_score(true_labels, predictions, average='weighted'))
        print("Recall: ", recall_score(true_labels, predictions, average='weighted'))
        print("F1-score: ", f1_score(true_labels, predictions, average='weighted'))
