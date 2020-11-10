# third party python libraries
import pandas as pd


def transform_datetime_column(input_df: pd.DataFrame) -> pd.DataFrame:
    """

    :param input_df: input dataframe with "datetime" column
    :return: a dateframe with new columns such as hour, month, weekday columns
    """
    input_df.loc[:, 'datetime'] = pd.to_datetime(input_df['datetime'])
    input_df.loc[:, "dayofweek"] = input_df["datetime"].dt.dayofweek
    input_df.loc[:, "month"] = input_df["datetime"].dt.month
    input_df.loc[:, "hour"] = input_df["datetime"].dt.hour
    input_df = input_df.drop(['datetime'], axis=1)

    return input_df
