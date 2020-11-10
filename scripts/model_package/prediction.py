# standard python libraries imports
import json
from typing import Any
import joblib

# third party python libraries
import pandas as pd
import numpy as np
from scripts.model_package.transform_input import transform_inputs


def predict_count(inputs: pd.DataFrame, loaded_model: Any) -> np.ndarray:
    """
    This function returns count number for inputs:
    ["datetime", "season", "holiday", "workingday",
    "weather", "temp", "atemp", "humidity", "windspeed"]
    :param inputs: dateframe object
    :param model_path: path to the trained model file
    :return: numpy arrays with predicted values
    """
    try:
        x_inputs = transform_inputs(inputs)
    except Exception as e:
        return "\n\n\nThe input data cannot be tranformed", type(inputs), inputs.shape,\
               inputs.columns, getattr(e, 'message', repr(e))
    try:
        predictions = loaded_model.predict(x_inputs)
    except Exception as e:
        return "\n\n\nModel did not predict any value \n\n\n", getattr(e, 'message', repr(e))

    # during traning phase, we used np.log transformation on the target column to
    # Now, the function np.exp needs to be applied on the predicted values
    predictions = np.exp(predictions)
    return np.round(predictions,0)


if __name__ == "__main__":
    MODEL_PATH = r'../../resources/final_model_rf.sav'
    LOADED_MODEL = joblib.load(MODEL_PATH)  # our trained model
    data_df = pd.read_csv(r"../../data/test.csv").iloc[:100, :]  # input to the model
    j_data = json.dumps(data_df.to_dict('list'))
    data = json.loads(j_data)
    input_data_df = pd.DataFrame.from_dict(data)
    y_hat = predict_count(input_data_df, LOADED_MODEL)
    print(y_hat[:])

