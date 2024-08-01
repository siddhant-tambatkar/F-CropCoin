from flask import Flask, render_template, request, jsonify
import requests
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
import traceback  # For detailed error logging

app = Flask(__name__)

# Reading the dataset
data = pd.read_csv('cpdata.csv')

# Assuming the dataset has the following columns:
# 'temperature', 'humidity', 'ph', 'rainfall', 'label'

# Separating features and target
X = data[['temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Creating dummy variable for target i.e label
label = pd.get_dummies(y).iloc[:, 1:]
data = pd.concat([X, label], axis=1)

# Splitting data into training and test sets
train, test = train_test_split(data, test_size=0.3, random_state=42)

X_train = train[['temperature', 'humidity', 'ph', 'rainfall']]
y_train = train.iloc[:, 4:].values

X_test = test[['temperature', 'humidity', 'ph', 'rainfall']]
y_test = test.iloc[:, 4:].values

# Standardizing features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing Decision Tree classifier
clf = DecisionTreeRegressor()

# Fitting the classifier into training set
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# Finding the accuracy of the model
a = accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
print("The accuracy of this model is: ", a * 100)

# Predefined list of crops
crops = ['wheat', 'mungbean', 'Tea', 'millet', 'maize', 'lentil', 'jute', 'cofee', 'cotton', 'ground nut',
         'peas', 'rubber', 'sugarcane', 'tobacco', 'kidney beans', 'moth beans', 'coconut', 'blackgram',
         'adzuki beans', 'pigeon peas', 'chick peas', 'banana', 'grapes', 'apple', 'mango', 'muskmelon',
         'orange', 'papaya', 'watermelon', 'pomegranate']

# OpenWeatherMap API setup
API_KEY = "a2d56e8d1c0a9d68183ec98d4fb3b160"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"

# Define seasons
SEASONS = [
    ('Winter', (1, 2, 12)),
    ('Summer', (3, 4, 5)),
    ('Rainy', (6, 7, 8)),
    ('Winter', (9, 10, 11))
]

def get_season(month):
    return next(season for season, months in SEASONS if month in months)

def months_to_next_season(current_month):
    current_season = get_season(current_month)
    current_season_index = next(i for i, (season, _) in enumerate(SEASONS) if season == current_season)
    next_season_index = (current_season_index + 1) % len(SEASONS)
    next_season_start = SEASONS[next_season_index][1][0]
    
    if next_season_start <= current_month:
        next_season_start += 12
    
    return next_season_start - current_month

def adjust_weather(temperature, humidity, months_to_next):
    if months_to_next > 4:
        return temperature, humidity
    
    current_season = get_season(datetime.now().month)
    next_season = get_season((datetime.now().month + months_to_next - 1) % 12 + 1)
    
    adjustment_factor = (5 - months_to_next) / 4  # 1/4, 1/2, 3/4, or 1
    
    if current_season == 'Winter' and next_season == 'Summer':
        temperature += 7 * adjustment_factor
        humidity -= 3 * adjustment_factor
    elif current_season == 'Summer' and next_season == 'Rainy':
        temperature -= 3 * adjustment_factor
        humidity += 7 * adjustment_factor
    elif current_season == 'Rainy' and next_season == 'Winter':
        temperature -= 7 * adjustment_factor
        humidity -= 3 * adjustment_factor
    
    return round(temperature, 2), round(humidity, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global sc  # Ensure sc is recognized as global

    if request.method == 'POST':
        try:
            # Get user inputs from form data
            city_name = request.form['city']
            ph = float(request.form['ph'])
            rain = float(request.form['rain'])

            # Get weather data from OpenWeatherMap API
            complete_url = f"{BASE_URL}appid={API_KEY}&q={city_name}"
            response = requests.get(complete_url)
            data = response.json()

            if data["cod"] != "404":
                main_data = data["main"]
                current_temperature = main_data["temp"] - 273.15  # Convert to Celsius
                current_humidity = main_data["humidity"]
                current_month = datetime.now().month
                months_to_next = months_to_next_season(current_month)
                adjusted_temp, adjusted_humidity = adjust_weather(current_temperature, current_humidity, months_to_next)

                # Prepare data for prediction
                predictcrop = [[adjusted_temp, adjusted_humidity, ph, rain]]
                predictcrop = sc.transform(predictcrop)  # Standardize the input
                predictions = clf.predict(predictcrop)

                # Get top 3 predicted crops
                top_n = 3
                top_indices = np.argsort(predictions[0])[-top_n:][::-1]
                predicted_crops = [crops[i] for i in top_indices]

                # Prepare response
                return jsonify({'predicted_crops': predicted_crops})
            else:
                return jsonify({'error': 'City Not Found'}), 404

        except Exception as e:
            traceback.print_exc()  # Print detailed exception traceback
            return jsonify({'error': 'An error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5003)
