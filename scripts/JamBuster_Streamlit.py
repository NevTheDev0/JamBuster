import streamlit as st
import joblib
from datetime import datetime, time  # ✅ We're using `time` from datetime
import pandas as pd
import requests
from traffic_logger import *
from prep_data import *

LAT = -5.135399
LON = 119.423790

roads = [
    {"name": "Boulevard", "lat": -5.156843, "lon": 119.446864},
    {"name": "Tun Abdul Razak", "lat": -5.180458, "lon": 119.465541},
    {"name": "Sultan Alauddin", "lat": -5.202323, "lon": 119.495785},
]


def is_rush_hour(ts):
    return (time(6, 0) <= ts.time() <= time(8, 0)) or (
        time(16, 0) <= ts.time() <= time(18, 0)
    )


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


def get_weather_at_time(timestamp):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": timestamp.date().isoformat(),
        "end_date": timestamp.date().isoformat(),
        "hourly": "temperature_2m,weathercode",
        "timezone": "Asia/Makassar",
    }

    r = requests.get(url, params=params)
    data = r.json()

    if "hourly" not in data:
        print(f"[DEBUG] No weather data for {timestamp.date()} (probably future date)")
        return "unknown"

    weather_df = pd.DataFrame(
        {
            "rounded_hour": pd.to_datetime(data["hourly"]["time"]),
            "temperature": data["hourly"]["temperature_2m"],
            "weathercode": data["hourly"]["weathercode"],
        }
    )

    weather_df["rounded_hour"] = weather_df["rounded_hour"].dt.floor("H")
    target_hour = pd.Timestamp(timestamp).floor("H")
    hour_match = weather_df[weather_df["rounded_hour"] == target_hour]

    if hour_match.empty:
        return "unknown"

    raw_code = hour_match["weathercode"].iloc[0]
    if pd.isna(raw_code) or raw_code is None:
        return "unknown"

    weather_code = int(raw_code)
    return decode_weather(weather_code)


def interpret_prediction(pred):
    if pred == 1:
        return "🚨 About to jam, I'd be careful"
    else:
        return "🍏 This road won't jam until a few minutes, dw"


def predict_traffic(road, selected_date, selected_time):
    Model = joblib.load("models/XGBClassifier.pkl")
    pipeline = joblib.load("models/preprocessor_pipeline.pkl")

    road_dict = next(r for r in roads if r["name"] == road)
    road_data = get_traffic_data(road_dict)

    current_speed = road_data["current_speed"]
    free_flow_speed = road_data["free_flow_speed"]
    confidence = road_data["confidence"]

    timestamp = datetime.combine(selected_date, selected_time)
    is_weekend = timestamp.weekday() >= 5
    rush_hour_flag = is_rush_hour(timestamp)
    weather = get_weather_at_time(timestamp)

    input_df = pd.DataFrame(
        [
            {
                "hour": timestamp.hour,
                "timestamp": timestamp,
                "is_weekend": is_weekend,
                "rush_hour": rush_hour_flag,
                "current_speed": current_speed,
                "free_flow_speed": free_flow_speed,
                "confidence": confidence,
                "low_confidence": confidence < 0.3,
                "speed_ratio": current_speed / free_flow_speed,
                "weather": weather,
            }
        ]
    )

    processed_input = pipeline.transform(input_df)
    prediction = Model.predict(processed_input)[0]
    return interpret_prediction(prediction)


# --- Streamlit UI ---
st.title("JamBuster - Real Time Traffic Predictor using AI")
selected_road = st.selectbox("Select a road", [r["name"] for r in roads])
selected_date = st.date_input("Select the date")
selected_time = st.time_input("Select the time")
if selected_date > datetime.today().date():
    st.warning(
        "⚠️ Weather data not available for future dates. Traffic prediction may be less accurate."
    )


if st.button("Predict Traffic"):
    prediction = predict_traffic(selected_road, selected_date, selected_time)
    st.success(f"Prediction: {prediction}")
    with st.expander("ℹ️ What do the predictions mean?"):
        st.markdown("""
        - 🍏 **Smooth**: Traffic is light, road's clear.
        - 🚨 **Jammed**: Traffic is building up, expect delays.
        """)
