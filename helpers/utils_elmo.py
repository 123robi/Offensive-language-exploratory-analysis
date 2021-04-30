import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras as keras
from keras.layers import Input, Lambda, Dense
from keras.models import Model

def ELMoEmbedding(x):
    embed = hub.Module("models/ELMo/module/module_elmo")
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

def build_model():
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
    dense = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(embedding)
    pred = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=[input_text], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model