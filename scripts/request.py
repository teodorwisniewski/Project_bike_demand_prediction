import pandas as pd
import requests
import json
import joblib
from scripts.model_package.prediction import predict_count

data_df = pd.read_csv(r"../data/train.csv").iloc[:15,:]# input to the model


checking = predict_count(data_df,joblib.load('../resources/final_model_rf.sav'))
j_data = data_df.to_dict('list')
# defining the header info for the api request
print("before requesting")
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

# making the api request
r = requests.post(url='http://127.0.0.1:5000/predictions/', data=json.dumps(j_data), headers=headers)
print(r)
print(r.text)
# getting the json data out
data = r.json()

# displaying the data
print(data)