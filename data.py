import pandas as pd
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

def time_to_float(time_str):
      # parse time string into datetime object
      dt_obj = datetime.strptime(time_str, '%H:%M')
      # extract hour and minute components
      hour = dt_obj.hour
      minute = dt_obj.minute
      # convert to float between 0 to 24
      time_float = hour + minute / 60
      return time_float

def handle_categorical(df):
  # sport type
  df = df.join(pd.get_dummies(df["Sport Type"]))
  df = df.drop("Sport Type", axis=1)
  # weekday
  df = df.join(pd.get_dummies(df["Weekday"]))
  df = df.drop("Weekday", axis=1)
  # start time
  df["Start Time in float"] = df["Start Time"].apply(time_to_float)
  df = df.drop("Start Time", axis=1)
  return df

def generate_input(weekday, sport_type, start_time, duration, calories_burned):
  df2 = pd.DataFrame(columns=df.columns)
  new_row = {weekday: 1,
          sport_type: 1,
          "Start Time in float": time_to_float(start_time),
          "Duration (min)": duration,
          "Calories Burned": calories_burned}
  df2 = df2.append(new_row, ignore_index=True)
  df2 = df2.fillna(0)
  return list(df2.iloc[0][1:])

df = pd.read_csv("sport_data.csv")
df = handle_categorical(df)

# fit model
X = df[list(df.columns)[1:]]
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)

query = [generate_input("Sunday", "Basketball", "16:00", 120, 1000)]
distances, indices = nbrs.kneighbors(query)
print(indices)
for index in indices:
  print(df.iloc[index])