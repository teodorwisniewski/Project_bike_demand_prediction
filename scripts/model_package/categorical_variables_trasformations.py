
import pandas as pd


def categ_vars_trasform(input_df: pd.DataFrame) -> pd.DataFrame:
    """

    :param input_df: input dataframe with categorical variables  "dayofweek",
     "month", "hour", "weather", "season"
    :return: a dateframe with new columns with dummy variables
    """
    category_cols_names = ["season", "weather", "holiday", "workingday", "hour", "dayofweek", "month"]
    for var in category_cols_names:
        input_df[var] = input_df[var].astype("category")

    columns_to_transform = ["dayofweek", "month", "hour", "weather", "season"]
    for col_to_transform in columns_to_transform:
        aux_df = pd.get_dummies(input_df[col_to_transform], prefix=col_to_transform + "_cat")
        input_df = pd.concat([input_df, aux_df], axis=1)

    return input_df


