import requests
import time
import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime


# API Set Up
load_dotenv()
api_key = os.getenv("TRAFFIC_API_KEY"), st.secrets.get("TRAFFIC_API_KEY") # Ignore this, this is for the Streamlit to use



roads = [
    {"name": "Boulevard", "lat": -5.156843, "lon": 119.446864},
    {"name": "Tun Abdul Razak", "lat": -5.180458, "lon": 119.465541},
    {"name": "Sultan Alauddin", "lat": -5.202323, "lon": 119.495785},
]


def get_traffic_data(road):
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/22/json?point={road['lat']},{road['lon']}&key={api_key}"
    response = requests.get(url)
    data = response.json()

    if "flowSegmentData" not in data:
        print(f"Error for {road['name']} missing or not found")

    segment = data["flowSegmentData"]

    return {
        "road": road["name"],
        "lat": road["lat"],
        "lon": road["lon"],
        "current_speed": segment["currentSpeed"],
        "free_flow_speed": segment["freeFlowSpeed"],
        "confidence": segment["confidence"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def collect_all_data():
    rows = []
    for road in roads:
        result = get_traffic_data(road)
        if result:
            rows.append(result)
    return rows


def start_logger():
    for i in range(1000):
        data = collect_all_data()
        if data:
            df = pd.DataFrame(data)
            df.to_csv(
                "traffic_log.csv",
                mode="a",
                header=not pd.io.common.file_exists("traffic_log.csv"),
                index=False,
            )
        print(f"Logged data at {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(300)


if __name__ == "__main__":
    start_logger()
