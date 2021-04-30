import pandas as pd
import numpy as np
import keras.backend as K
from helpers.utils_elmo import *

test_set = pd.read_csv("data/datasets/test.csv", encoding='utf-8')
X = test_set['text']
y = test_set['hatespeech']

num_correct = 0
model_elmo = build_model()
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model_elmo.load_weights('models/ELMo/model_elmo_weights.h5')
    import time
    t = time.time()
    for i in range(len(X)):
        predicts = model_elmo.predict(np.array([str(X[i]), str(X[i])], dtype=object)[:, np.newaxis])
        if (predicts[0] > 0.5 and y[i]) or (predicts[0] <= 0.5 and not y[i]):
            num_correct += 1
    print("Correct: ", num_correct)
    print("All: ", len(X))
    print("Acc: ", num_correct/len(X))
    print("time: ", time.time() - t)
