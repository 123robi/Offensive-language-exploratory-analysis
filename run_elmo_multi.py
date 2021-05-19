import pandas as pd
import numpy as np
import keras.backend as K
from helpers.utils_elmo import *

test_set = pd.read_csv("data/transformed_datasets/nova24_multi.csv", encoding='utf-8')
X = test_set['comment']
y = test_set['type']

model_elmo = build_model()

predictions = []
true_labels = []
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model_elmo.load_weights('models/ELMo/model_elmo_weights_multi.h5')
    import time
    t = time.time()
    for i in range(len(X)):
        predicts = model_elmo.predict(np.array([str(X[i]), str(X[i])], dtype=object)[:, np.newaxis])
        predictions.append(np.argmax(predicts[0]))
        true_labels.append(y[i])


    acc = np.sum(np.array(predictions) == np.array(true_labels)) / len(predictions)
    print(acc)

