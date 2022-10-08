import pytest
import pandas as pd 
import numpy as np
import collections
    

def test_reading_data():

    df = pd.read_csv("./data/sentences.csv")
    assert df.shape[0] > 0
    assert df.shape[1] > 0
    assert df.shape[0] == 10341812
    assert df.shape[1] == 3

def test_columns_names():
    df = pd.read_csv("./data/sentences.csv")
    columns_names = df.columns 
    assert collections.Counter(columns_names) == collections.Counter(["id", "lan_code", "sentence"])


def test_number_languages():

    df = pd.read_csv("./data/sentences.csv")
    df_grp = df.lan_code.value_counts()
    assert df_grp.index[0] == 'eng'
    assert df_grp.index[1] == 'rus'
    assert df_grp.index[2] == 'ita'
    assert df_grp.index[3] == 'tur'
    assert df_grp.index[4] == 'epo'

    assert df_grp[0] == 1586621
    assert df_grp[1] == 909951
    assert df_grp[2] == 805104
    assert df_grp[3] == 717897
