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

# def time_to_float(time_str):
#       # parse time string into datetime object
#       dt_obj = datetime.strptime(time_str, '%H:%M')
#       # extract hour and minute components
#       hour = dt_obj.hour
#       minute = dt_obj.minute
#       # convert to float between 0 to 24
#       time_float = hour + minute / 60
#       return time_float


def handle_categorical(df):
    # sport type
    df = df.join(pd.get_dummies(df["Sport Type"]))
    df = df.drop("Sport Type", axis=1)
    # weekday
    df = df.join(pd.get_dummies(df["Weekday"]))
    df = df.drop("Weekday", axis=1)
    # start time
    # df["Start Time in float"] = df["Start Time"].apply(time_to_float)
    # df = df.drop("Start Time", axis=1)
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


# df = pd.read_csv("sport_data.csv")
# # drop location
# df = df.drop('Location (City)', axis=1)
# df = handle_categorical(df)
# df = merge_user_rows(df)

# # fit model
# X = df[list(df.columns)]
# neigh = NearestNeighbors(n_neighbors=2)
# nbrs = neigh.fit(X)

# query = [
#     [11, "Monday", "Basketball", 120, 800],
#     [11, "Sunday", "Running", 60, 400]
# ]
# input_df = generate_input(query)
# processed_input = merge_user_rows(input_df).iloc[:, 1:].values.tolist()
# print(processed_input)

# distances, indices = nbrs.kneighbors(processed_input)
# print(indices)
# usernames = []
# for index in indices[0]:
#     print(df.iloc[index])
