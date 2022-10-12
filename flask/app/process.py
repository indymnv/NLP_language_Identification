import numpy as np
import os
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow import keras
from keras import Sequential

# Hyperparams if GPU is available
if tf.test.is_gpu_available():
    # GPU
    BATCH_SIZE = 512
    EPOCHS = 12
    
# Hyperparams for CPU training
else:
    # CPU
    BATCH_SIZE = 64
    EPOCHS = 1
NUM_WORDS = 100000
MAX_LEN = 1000
NUM_CLASSES = 2

def builtModel(vocad_dim = NUM_WORDS, max_len = MAX_LEN, num_classes = NUM_CLASSES):
    model = Sequential()
    
    ## ADD LAYERS
    model.add(layers.Embedding(input_dim=vocad_dim,
                              output_dim=32,
                              input_length=max_len))
    model.add(layers.LSTM(units=32,
                         return_sequences=False,
                         activation='relu'))
    model.add(layers.Dense(units=1,
                          activation="sigmoid"))
    
    #compiler
    model.compile(loss='binary_crossentropy',
                  optimizer ='adam',
                  metrics=['accuracy'])
    return model

def init():
    with open('/Users/pitehrhurtadocayo/Documents/DevOps/Project_0006_NLP/NLP_LanguageIdentification/models/model.json') as json_file:
        json_config = json_file.read()
    nlpModel = keras.models.model_from_json(json_config)
    nlpModel.load_weights('/Users/pitehrhurtadocayo/Documents/DevOps/Project_0006_NLP/NLP_LanguageIdentification/models/model.h5')
    #nlpModel.summary()

    #for layer in nlpModel.layers:
    #    print(layer.name, layer)

    return nlpModel.layers[2].weights