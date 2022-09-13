import numpy as np
import pandas as pd
from dataset import load_data, generate_dataset
import gc
from keras import backend
from KTFCM import generate_points


def function(x):
    print(x)
    return (1, 2, 3)


if __name__ == '__main__':
    print(type(function(2)))
