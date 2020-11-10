# standard library imports
import joblib
import sys
import traceback

# third party libraries imports
from flask import Flask, jsonify, request, render_template
import pandas as pd

# projects packages imports
from scripts.model_package.prediction import predict_count


MODEL_PATH = 'resources/final_model_rf.sav'
LOADED_MODEL = joblib.load(MODEL_PATH) # our trained model
app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello():
    return jsonify("Forecast bike rental demand")


@app.route("/predictions/", methods=['POST'])
def predictions():
    '''
    For direct API calls trought request
    '''
    print("I entered to predictions")
    try:
        data = request.get_json()
        output_keys = data["datetime"]
        input_data_df = pd.DataFrame.from_dict(data)
        y_hat = predict_count(input_data_df, LOADED_MODEL)
    except Exception as e:
        return jsonify({'trace': traceback.format_exc(),"error message": getattr(e, 'message', repr(e)) })
    print("done")
    output_dict = {key:int(val) for key,val in zip(output_keys,y_hat.tolist())}
    result = {"Predicted values": output_dict}
    return jsonify(result)





if __name__ == "__main__":
    # TEST_DF = pd.read_csv(r"data/test.csv")  # input to the model
    # predictions_loaded = predict_count(TEST_DF, LOADED_MODEL)
    # print(type(LOADED_MODEL))
    # print(predictions_loaded)
    app.run(debug=True)