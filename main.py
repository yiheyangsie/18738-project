import streamlit as st
import numpy as np
from data import *

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from data import *

from cities import city_names


st.title("Hey bro, need a sport mate?")

st.write("""
# Explore your city, sport type, duration, and calories goal
Which fits you best?
""")

city_name = st.sidebar.selectbox(
    'Select City',
    city_names
)

st.write(f"## {city_name} people")

weekday = st.sidebar.selectbox(
    'Select Weekday',
    ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
)

sport_type = st.sidebar.selectbox(
    'Select Sport Type',
    ('Swimming', 'Running', 'Yoga', 'Basketball', 'Cycling', 'Tennis',
     'Walking', 'Stretching', 'Calisthenics', 'Soccer', 'Table Tennis')
)

duration = st.sidebar.selectbox(
    'Select duration(min)',
    (5, 10, 15, 20, 25, 30, 35, 40, 45,
     50, 55, 60, 65, 70, 75, 80, 85, 90)
)
calories = st.sidebar.selectbox(
    'Select calories goal',
    (50, 100, 150, 200,
     250, 300, 350, 400,
     450, 500, 550, 600
     )
)


st.write('City:', city_name)
st.write('Weekday:', weekday)

st.write('Sport type:', sport_type)
st.write('Duration:', duration)
st.write('Calories goal:', calories)


df = pd.read_csv("sport_data2.csv")
# drop location
df = df.drop('Location (City)', axis=1)
df = handle_categorical(df)
duration_min = df["Duration (min)"].min()
duration_max = df["Duration (min)"].max()
calories_min = df["Calories Consumed"].min()
calories_max = df["Calories Consumed"].max()
df = merge_user_rows(df, duration_min, duration_max,
                     calories_min,  calories_max)


# fit model
X = df[list(df.columns)[1:]]

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)

query = [
    [11, weekday, sport_type, duration, calories]
]
input_df = generate_input(query, df)
processed_input = merge_user_rows(
    input_df, duration_min, duration_max, calories_min,  calories_max).iloc[:, 1:].values.tolist()

print(f"processed_input: {processed_input}")

distances, indices = nbrs.kneighbors(processed_input)
print(f"indices: {indices}")
usernames = []
for index in indices[0]:
    usernames.append(df.iloc[index, 0])
print(f"usernames: {usernames}")

output_users = ""
for user in usernames:
    output_users += user
    output_users += ","
st.write('Mates selected:', output_users)


print("it renders again")
