# Testing combs,permutations,factorials

# our own packages
from scripts.model_package.transform_input import transform_inputs

# standard library imports
import joblib
import os

# third party imports
import pytest
import numpy as np
import pandas as pd
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
X_EXPECTED = pd.read_csv(r"../data/expected_input_data.csv") # input to the model
Y_EXPECTED = pd.read_csv(r"../data/expected_output_data.csv", header =0)["count"] # input to the model
filename_model = '../resources/final_model_rf.sav'
LOADED_MODEL = joblib.load(filename_model) # our trained model
TEST_DF = pd.read_csv(r"../data/test.csv") # input to the model
TRAIN_DF = pd.read_csv(r"../data/train.csv") # input to the model
TRAIN_DF = TRAIN_DF.drop(["casual", "registered", "count"], axis = 1)


def test_shape_of_input():
    """
    Checking whether or not an input array of the model has 56 features/columns
    :return: test
    """
    assert X_EXPECTED.shape[1] == 56
    x_train = transform_inputs(TRAIN_DF)
    assert x_train.shape[1] == 56
    x_test = transform_inputs(TEST_DF)
    assert x_test.shape[1] == 56
    x_train = transform_inputs(TRAIN_DF.iloc[:5,:])
    assert x_train.shape[1] == 56



def test_input_values_format():
    """
    Checking whether or not an input array of the model has 56 features/columns
    :return: test
    """
    x_train = transform_inputs(TRAIN_DF)
    assert all(x_train.columns == X_EXPECTED.columns)
    assert np.allclose(x_train.values, X_EXPECTED.values)
    data = pd.read_json(r'../data/train_df_inputs_json.json')
    out = transform_inputs(data)
    assert np.allclose(out.values,X_EXPECTED.values)
    x_train = transform_inputs(TRAIN_DF.iloc[:5,:])
    assert np.allclose(x_train.values, X_EXPECTED.iloc[:5,:].values)
    #np.transpose(np.where(~(x_train.values== X_EXPECTED.iloc[:5,:].values)))



def test_simple_prediction():
    """
    Note: results may vary given the stochastic nature of the algorithm or evaluation procedure,
    or differences in numerical precision. Consider running the example a few times
    and compare the average outcome
    """
    x_train = transform_inputs(TRAIN_DF)
    predictions_loaded = LOADED_MODEL.predict(x_train)
    expected_array = np.array([3.17660787, 3.57341024, 3.38910521, 2.71681494, 1.17841518])
    obtained_array = predictions_loaded[:5]
    assert np.allclose(obtained_array, expected_array, rtol=0.5)
    predictions_loaded = LOADED_MODEL.predict(x_train.iloc[:5,:])
    assert np.allclose(predictions_loaded, expected_array, rtol=0.5)
    predictions_loaded = LOADED_MODEL.predict(x_train.iloc[:1,:])
    assert np.allclose(predictions_loaded, expected_array[0], rtol=0.5)