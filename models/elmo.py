import pandas as pd
import keras.backend as K
import helpers.preprocessing as pp
from helpers.utils_elmo import *

embed = hub.Module("models/ELMo/module/module_elmo")

def preprocess_text(text):
    #text = pp.remove_stopwords(text, "")
    text = pp.remove_mentions(text)
    text = pp.remove_additional_spaces(text)
    return pp.keep_chars_only(text)

model_elmo = build_model()
model_elmo.summary()

train_set = pd.read_csv("data/datasets/train.csv", encoding='utf-8')
X = train_set['text']
y = train_set['hatespeech']
#TODO preprocess text
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    history = model_elmo.fit(X, y, epochs=20, batch_size=16, validation_split = 0.2)
    model_elmo.save_weights('models/ELMo/model_elmo_weights2.h5')

