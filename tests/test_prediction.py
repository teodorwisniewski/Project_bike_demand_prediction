# Testing combs,permutations,factorials

# our own packages
from scripts.model_package.transform_input import transform_inputs
from scripts.model_package.prediction import predict_count


# standard library imports
import joblib
import os

# third party imports
import pytest
import numpy as np
import pandas as pd
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
Y_EXPECTED = pd.read_csv(r"../data/expected_output_data.csv", header =0)["count"] # input to the model
MODEL_PATH = '../resources/final_model_rf.sav'
LOADED_MODEL = joblib.load(MODEL_PATH) # our trained model
TRAIN_DF = pd.read_csv(r"../data/train.csv") # input to the model
TRAIN_DF = TRAIN_DF.drop(["casual", "registered", "count"], axis = 1)


def test_predict_count():
    """
    Note: results may vary given the stochastic nature of the algorithm or evaluation procedure,
    or differences in numerical precision. Consider running the example a few times
    and compare the average outcome
    """
    predictions_loaded = predict_count(TRAIN_DF, LOADED_MODEL)
    expected_array = np.exp(np.array([3.17660787, 3.57341024, 3.38910521, 2.71681494, 1.17841518]))
    obtained_array = predictions_loaded[:5]
    assert np.allclose(obtained_array, expected_array, rtol=0.5)


def rmsl_error_metric(y, y_hat) -> float:
    """
    # Root Mean Squared Logarithmic Error (RMSLE)
    :param y: array-like structure actual values to predict
    :param y_hat: array-like structure   predicted values
    :return: float
    """
    y = np.exp(y)
    y_hat = np.exp(y_hat)
    log1 = np.nan_to_num(np.array([np.log(obs + 1) for obs in y]))
    log2 = np.nan_to_num(np.array([np.log(obs + 1) for obs in y_hat]))
    res = (log1 - log2) ** 2
    return np.sqrt(np.mean(res))


def test_rmsl_error_metric():
    predictions_loaded = predict_count(TRAIN_DF, LOADED_MODEL)
    assert np.isclose(0.13745188585638188,rmsl_error_metric(Y_EXPECTED, np.log1p(predictions_loaded)), rtol=5e-01)