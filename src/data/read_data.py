"""
Functions for reading and preparing the MNIST data.
"""
import sys, os
import numpy as np
import pandas as pd

def read_data_reshaped(input_filepath):
    """
    Reads data from data/raw into csv format and converts it from wide format (one column per pixel) to the expected 28x28 pixel images with one colour channel (grayscale).
    """
    raw_data = pd.read_csv(input_filepath)
    y = np.array(raw_data["label"])
    X = raw_data.drop("label", axis=1)
    X = X.values.reshape(-1, 28, 28, 1)
    print("Loaded {} images.".format(len(X)))
    return X, y
