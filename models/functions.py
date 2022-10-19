import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(chosed_lan=None, DEBUG = False):
  path = "./models/data/sentences.csv"
  df = pd.read_csv(path)

  # Drop innecessary column ID
  df.drop(columns=["id"],inplace=True)

  # Choiced lan
  if chosed_lan is not None:
    df = df[df["lan_code"].isin(chosed_lan)]

  if DEBUG:
    print(df.describe())
    df_group = df.groupby("lan_code").count().sort_values("sentence",ascending=False)
    print("#"*50, " RESULT:")
    print(df_group)
  
  return df

def sampling_dataset(dataframe, balanced = False):
  # falta desarrolloo

  return False

def pre_encoding(dataframe, DEBUG = False):
  if DEBUG:
    print(dataframe.head(2))

  classes = list(dataframe['lan_code'].unique())
  cls_to_num = {
    cls : i
    for i,cls in enumerate(classes)
  }

  num_to_cls = {
      i : cls
      for cls,i in cls_to_num.items()
  }

  dataframe["lan_code"] = dataframe["lan_code"].map(cls_to_num).astype(int)

  if DEBUG:
    print(dataframe.head(2))
    print("ENCODING LAN: ")
    print(cls_to_num)
  
  return dataframe, cls_to_num ,num_to_cls

def split_training_testing(dataframe, DEBUG=False):
  X = dataframe.pop('sentence').values
  y = dataframe.copy().values.T[0]

  if DEBUG:
    print("Phrase and LANCODE:")
    print(X[:2])
    print(y[:2])

  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      train_size=0.8,
                                                      random_state=50,
                                                      shuffle=True)
  if DEBUG:
    print("X train: " +str(len(X_train)))
    print("y train: " +str(len(y_train)))
    print("X test: " +str(len(X_test)))
    print("y test: " +str(len(y_test)))
  return X_train,X_test,y_train,y_test