# Import LabelEncoder
import pandas as pd

# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix

# creating labelEncoder for normalising the label
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
df = pd.read_csv("weather dataset.csv")
"""Encoding is done in an alphabetical order"""
weather_encoded = le.fit_transform(df.weather)
print("WEATHER ENCODE : ", weather_encoded)

temprature_encoded = le.fit_transform(df.temp)
print("TEMPERATURE ENCODE : ", temprature_encoded)

play_encoded = le.fit_transform(df.play)
print("PLAY ENCODE : ", play_encoded)

# Combining weather and temp into single list of tuples
features = zip(weather_encoded, temprature_encoded)
features = list(features)
print(list(features))

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features, play_encoded)

# Predict Output
predicted = []
for weather, temprature in zip(weather_encoded, temprature_encoded):
    predicted.append(model.predict([[weather, temprature]]))  # 0:Overcast, 2:Mild

print(accuracy_score(y_true=play_encoded, y_pred=predicted))
print(confusion_matrix(play_encoded, predicted))
