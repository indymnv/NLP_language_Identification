import pytest
import pandas as pd 
import numpy as np


def test_reading_data():
    df = pd.read_csv("./data/sentences.csv")

    assert df.shape[0] > 0
    assert df.shape[1] > 0