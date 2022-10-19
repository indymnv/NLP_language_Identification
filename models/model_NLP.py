import tensorflow as tf
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras import Sequential

import pandas as pd
import numpy as np

NUM_WORDS = 100000
MAX_LEN = 100
NUM_CLASSES = 2


def tokenize_and_sequence(train_sentences, test_sentences, NUM_WORDS = NUM_WORDS, MAX_LEN = MAX_LEN, NUM_CLASSES = NUM_CLASSES, DEBUG = False):
  tok = Tokenizer(num_words = NUM_WORDS,
                    split = ' ',
                    oov_token='<OOV>')
  tok.fit_on_texts(train_sentences)

  # training set
  train_seq = tok.texts_to_sequences(train_sentences)
  #max_len=max([len(i) for i in train_seq])
  train_seq = pad_sequences(
                    train_seq,
                    padding = 'post',
                    maxlen = MAX_LEN,
                    truncating = 'post'
                )

  # testing set
  test_seq = tok.texts_to_sequences(test_sentences)
  test_seq = pad_sequences(
                    test_seq,
                    padding = 'post',
                    maxlen = MAX_LEN,
                    truncating = 'post'
                )
  if DEBUG:
      print("PADD TRAIN:")
      print(train_sentences[:2])
      print(train_seq[:2])
      print("PADD TEST:")
      print(test_sentences[:2])
      print(test_seq[:2])
      print("-"*100)
      print("TOK: --info--")
      #df = pd.DataFrame([{"frase": word, "count": value} for word, value in tok.get_config()['word_counts'].items()])
      #df.to_csv('.output/words.csv',index=False)
      #print(df.head())
      #rint(type(tok.get_config()['word_counts']))

  return train_seq, test_seq, tok

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


def evaluate_model(model, X_testing, Y_testing, BATCH_SIZE = 12):
  y_pred = model.predict(X_testing, batch_size=BATCH_SIZE, verbose=1)
  y_pred = y_pred.argmax(axis=1)
  acc = accuracy_score(Y_testing, y_pred)
  conf_mat = confusion_matrix(Y_testing, y_pred)
  conf_mat_recall = confusion_matrix(Y_testing, y_pred, normalize='true')
  conf_mat_precision = confusion_matrix(Y_testing, y_pred, normalize='pred')
  print(f"Accuracy = {acc:.2%}")
  return y_pred

def predict(model, input):

  return False