# Third party imports
import numpy as np
import pandas as pd
import json

# project's moduls imports
from scripts.model_package.datetime_column_tranformations import transform_datetime_column
from scripts.model_package.categorical_variables_trasformations import categ_vars_trasform


def transform_inputs(inputs_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function should carry out all feature engineering tasks such as
    - extracting additional variables
    - dropping unnessary data
    - add dummy variables for categorical variables
    In the end, the dataframe object with 56 columns will be created for regressor model
    :param inputs_df: numpy array or pandas dataframe
    :return: dataframe objects with 56 columns
    """
    # This list is necessary ensure that our inputs dataframe will have a good order of columns
    columns_order = ['holiday', 'workingday', 'temp', 'humidity', 'windspeed', 'dayofweek_cat_0',
                     'dayofweek_cat_1', 'dayofweek_cat_2', 'dayofweek_cat_3', 'dayofweek_cat_4', 'dayofweek_cat_5',
                     'dayofweek_cat_6', 'month_cat_1', 'month_cat_2', 'month_cat_3', 'month_cat_4', 'month_cat_5',
                     'month_cat_6', 'month_cat_7', 'month_cat_8', 'month_cat_9', 'month_cat_10', 'month_cat_11',
                     'month_cat_12', 'hour_cat_0', 'hour_cat_1', 'hour_cat_2', 'hour_cat_3', 'hour_cat_4',
                     'hour_cat_5', 'hour_cat_6', 'hour_cat_7', 'hour_cat_8', 'hour_cat_9', 'hour_cat_10',
                     'hour_cat_11', 'hour_cat_12', 'hour_cat_13', 'hour_cat_14', 'hour_cat_15', 'hour_cat_16',
                     'hour_cat_17', 'hour_cat_18', 'hour_cat_19', 'hour_cat_20', 'hour_cat_21', 'hour_cat_22',
                     'hour_cat_23', 'weather_cat_1', 'weather_cat_2', 'weather_cat_3', 'weather_cat_4', 'season_cat_1',
                     'season_cat_2', 'season_cat_3', 'season_cat_4']

    columns = ["datetime", "season", "holiday", "workingday",
               "weather", "temp", "atemp", "humidity", "windspeed"]
    if isinstance(inputs_df, (np.ndarray,)):
        inputs_df = pd.DataFrame(data=inputs_df[:, 0:len(columns)], columns=columns)

    inputs_df = inputs_df[columns]
    inputs_df = transform_datetime_column(inputs_df)
    inputs_df = categ_vars_trasform(inputs_df)
    columns_to_drop = ["atemp", "dayofweek", "month", "hour", "weather", "season"]
    inputs_df = inputs_df.drop(columns_to_drop, axis=1)
    if inputs_df.shape[1] != 56:

        empty_df = pd.DataFrame(columns=columns_order)
        inputs_df = pd.merge(empty_df, inputs_df, how="outer").replace(np.nan, 0)

    return inputs_df[columns_order]


if __name__ == "__main__":
    data = pd.read_json(r'../../data/train_df_inputs_json.json').iloc[:2,:]
    out = transform_inputs(data)
    print(out.shape)
