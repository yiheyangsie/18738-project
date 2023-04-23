import pandas as pd
from datetime import datetime
from sklearn.neighbors import NearestNeighbors


def merge_user_rows(user_df, duration_min, duration_max, calories_min,  calories_max):
    id_counts = dict(user_df['ID'].value_counts())
    merged_df = user_df.groupby('ID').sum().reset_index()
    merged_df['Duration (min)'] = merged_df.apply(
        lambda x: x['Duration (min)'] / id_counts[x['ID']], axis=1)
    merged_df['Calories Consumed'] = merged_df.apply(
        lambda x: x['Calories Consumed'] / id_counts[x['ID']], axis=1)
    # print(f"duration_range: {duration_range}")
    # print(f"calories_range: {calories_range}")
    merged_df["Duration (min)"] = (merged_df["Duration (min)"] - duration_min
                                   ) / (duration_max - duration_min)
    merged_df["Calories Consumed"] = (merged_df["Calories Consumed"] - calories_min) / (
        calories_max - calories_min)
    print(duration_max - duration_min)
    print(calories_max - calories_min)
    return merged_df


def handle_categorical(df):
    # sport type
    df = df.join(pd.get_dummies(df["Sport Type"]))
    df = df.drop("Sport Type", axis=1)
    # weekday
    df = df.join(pd.get_dummies(df["Weekday"]))
    df = df.drop("Weekday", axis=1)

    return df


def generate_input(user_info, df):
    df2 = pd.DataFrame(columns=df.columns)
    row_num = len(user_info)
    for row in user_info:
        new_row = {
            "ID": row[0],
            row[1]: 1,
            row[2]: 1,
            "Duration (min)": row[3],
            "Calories Consumed": row[4]}
        df2 = df2.append(new_row, ignore_index=True)
        df2 = df2.fillna(0)
    return df2
