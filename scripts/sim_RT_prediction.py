import pandas as pd
import joblib
import time
import requests


model = joblib.load("models/XGGBClassifier.pkl")
preprocessor = joblib.load("models/preprocessor_pipeline.pkl")

def load_latest_data(csv_path="traffic_log.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="timestamp", ascending=False).groupby("road").head(1)
    return df

def get_current_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": -5.135399,
        "longitude": 119.423790,
        "current_weather": True,
    }
    r = requests.get(url, params=params)
    data = r.json()
    code = data['current_weather']['weathercode']
    return decode_weather(code)

def decode_weather(code):
    if code in [0, 1]:
        return "clear"
    elif code in [2, 3]:
        return "cloudy"
    elif code in [45, 48]:
        return "foggy"
    elif code in [51, 53, 55, 56, 57]:
        return "drizzle"
    elif code in [61, 63, 65, 66, 67, 80, 81, 82]:
        return "rainy"
    elif code in [71, 73, 75, 77, 85, 86]:
        return "snowy"
    elif code in [95, 96, 99]:
        return "thunderstorm"
    else:
        return "unknown"


def add_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['current_speed']
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = df['timestamp'].dt.weekday >= 5
    df['rush_hour'] = df["hour"].apply(lambda h: h in range(7,10) or h in range(16,19))
    df['low_confidence'] = df["confidence"] < 0.5
    df['weather'] = get_current_weather()
    return df

def predict_live():
    raw_data = load_latest_data()
    featured = add_features(raw_data)
    X = preprocessor.transform(featured)
    predictions = model.predict(X)
    raw_data['predicted_congestions'] = predictions
    return raw_data[['road', 'timestamp', 'predicted_congestions']]

if __name__ == "__main__":
    while True:
        result = predict_live()
        print(result)
        time.sleep(310)  #refreshes every 5 minute and 10 seconds(to avoid clashing between running the logger and making a prediction)