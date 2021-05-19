import pandas as pd
import numpy as np
import keras.backend as K
from helpers.utils_elmo import *

test_set = pd.read_csv("data/datasets/test.csv", encoding='utf-8')
X = test_set['text']
y = test_set['hatespeech']

num_correct = 0
model_elmo = build_model()
TP = 0
TN = 0
FP = 0
FN = 0
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model_elmo.load_weights('models/ELMo/model_elmo_weights2.h5')
    import time
    t = time.time()
    for i in range(len(X)):
        predicts = model_elmo.predict(np.array([str(X[i]), str(X[i])], dtype=object)[:, np.newaxis])
        if (predicts[0] > 0.5 and y[i]):
            TP += 1
        elif (predicts[0] <= 0.5 and not y[i]):
            TN += 1
        elif (predicts[0] > 0.5 and not y[i]):
            FP += 1
        elif (predicts[0] < 0.5 and y[i]):
            FN += 1

    print("Accuracy: ", (TP + TN) / (TP + TN + FP + FN))
    print("Precision: ", (TP) / (TP + FP))
    print("Recall: ", (TP) / (TP + FN))
    print("F1-score:", (2*TP) / (2*TP + 2*FP + 2*FN))

    print("time: ", time.time() - t)
