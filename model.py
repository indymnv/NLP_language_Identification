### TO DEBUGGIN'
_DEBUG = True

# to load and manipulate data
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

# to make neural netowrk model, sequencing and tokenizing
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

# to evaluate and print results
from sklearn.metrics import  confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import squarify

## print paths
for root, dirs, files in os.walk('./data'):
    for filename in files:
        print(os.path.join(root,filename))


## vamos a medir el TIME de esta shit

df = pd.read_csv("./data/sentences.csv", nrows = 10_000)

## eliminar columna maldita que esta de mas ID
df.drop(columns=['id'],inplace=True)

## [x] no hay sentencias sin code - :D!
if _DEBUG:
    print(df.describe())

if _DEBUG:
    df_exploration = df.groupby("lan_code").count().sort_values("sentence",ascending = False)

    #!# posiblemente cuando se entrene el modelo los de baja frecuencia no va a predecir correctamente
    #!# como posible opcion 1 analizar con todo y luego ver el impacto de remover esos code lan con freq 1
    
    count_freq_1 = df_exploration[df_exploration.sentence<=10].sum()
    print("Code LANGUAGE with less frequency than 10: {}".format(count_freq_1))
    
    ## GRAPH - TREEMAP
    sizes_treemap = df_exploration["sentence"].tolist()
    labels_treemap = df_exploration.index.values.tolist()
    
    plt.figure(figsize=(10,10))
    squarify.plot(sizes = sizes_treemap, label = labels_treemap, alpha=0.2)
    plt.savefig('block_chart.jpeg')
    
    del df_exploration

# ENCODING CLASSES (cod language)
classes_list = list(df.lan_code.unique())
print("Count of Classes: {}".format(len(classes_list)))

## crearemos dos dic para usos posteriores de transformacion

cls_to_num = {
    cls : i
    for i,cls in enumerate(classes_list)
}

num_to_cls = {
    i : cls
    for cls,i in cls_to_num.items()
}

## cambiar el string-code por int-code para ingresarlos al modelo

print(df.head())
df.lan_code = df.lan_code.map(cls_to_num).astype(int)
print(df.head())

# Lista de sentencias
X = df.pop('sentence').values

# Lista de CODE LAN 
y = df.copy().values.T[0]

if _DEBUG:
    print(X)
    print(y)

## liberamos memoria virtual (RAM)
del df

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    train_size=0.8,
    random_state=50,
    shuffle = True
)

## liberamos memoria virtual (RAM)
del X, y

## parametros

NUM_WORDS = 1000000
MAX_LEN = 100
NUM_CLASSES = 404 # esto lo sacamos de arriba

def tokenize_and_sequence(
                            train_sentences,
                            test_sentences
                            ):
    tok = Tokenizer(num_words = NUM_WORDS, oov_token='<OOV>')
    tok.fit_on_texts(train_sentences)
    
    # training set
    train_seq = tok.texts_to_sequences(train_sentences)
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
    if _DEBUG:
        print(tok.get_config())
    
    return train_seq, test_seq, tok

## call function 
X_train, X_test, tok = tokenize_and_sequence(X_train, X_test)

# WARNING: problemas de tiempo de respuesta, se debe cambiar la config del notebook


if _DEBUG:
    print(X_train.shape)
    print(X_test.shape)

## heredamos la clases tf.keras.model (Classic)
class NLPModel(tf.keras.Model):
    def __init__(self,
                vocad_dim = NUM_WORDS,
                max_len = MAX_LEN,
                num_classes = NUM_CLASSES):
        super().__init__()
        
        self.embedding = layers.Embedding(input_dim = vocad_dim, output_dim = 32, input_length = max_len)
        
        self.lstm1 = layers.Bidirectional(layers.LSTM(units = 32, return_sequences=True))
        self.lstm2 = layers.Bidirectional(layers.LSTM(16))
        
        self.dense = layers.Dense(64, activation = 'relu')
        
        self.dropout = layers.Dropout(0.5)
        
        self.classifier = layers.Dense(num_classes, activation = 'softmax')
        
    @tf.function()
    def call(self, inputs, training = False):
        x = self.embedding(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense(x)
        if training:
            x = self.dropout(x, training = training)
        return self.classifier(x)


model = NLPModel()

## Especificamos la configuración de entrenamiento (optimizador, pérdida, métricas):
# uso de ADAM Algorithm

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=512, 
                    epochs=5
)


## datos del entrenamiento
history.history


y_pred = model.predict(X_test, batch_size=512, verbose=1)
y_pred = y_pred.argmax(axis=1)

acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
conf_mat_recall = confusion_matrix(y_test, y_pred, normalize='true')
conf_mat_precision = confusion_matrix(y_test, y_pred, normalize='pred')

print(f"Accuracy = {acc:.2%}")

plt.figure(figsize=(12, 12))
ax = sns.heatmap(conf_mat_precision, vmin=0, vmax=1)
cbar = ax.collections[0].colorbar
cbar.set_ticks([0, .25, 0.5, .75, 1])
cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
plt.savefig('heatmap-result.jpeg')

model = NLPModel()
#model.labels = tf.saved_model.Asset('./models/nlp.txt')
tf.saved_model.save(model, './models/')