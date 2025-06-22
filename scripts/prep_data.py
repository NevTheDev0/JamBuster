import pandas as pd
import requests
from time import sleep
from tqdm import tqdm
from datetime import time


#LAT LON for the city of Makassar
LAT = -5.135399
LON = 119.423790

#Reads the data pulled from traffic_logger.py
df = pd.read_csv("traffic_log.csv") 
df['timestamp'] = pd.to_datetime(df['timestamp'])


#Define and engineer new Features
df['congestion_percent'] = (1- df['current_speed'] / df['free_flow_speed']) * 100

def categorize_data(p):
    if p < 30:
        return 'Low'
    elif p < 60:
        return 'Moderate'
    else:
        return 'Severe'

def is_rush_hour(t):
    return (time(6, 0) <= t <= time(8, 0)) or (time(16, 0) <= t <= time(18, 0))

#Concat/add features to the original dataset
df['rush_hour'] = df['timestamp'].dt.time.apply(is_rush_hour)
df['congestion_level'] = df['congestion_percent'].apply(categorize_data)
df['low_confidence'] = df['confidence'] < 0.3
df['is_weekend'] = df['timestamp'].dt.weekday >= 5
df['hour'] = df['timestamp'].dt.hour
df['rounded_hour'] = df['timestamp'].dt.floor('H')
batched_hours = df['rounded_hour'].drop_duplicates().sort_values()
df['speed_diff'] = df['current_speed'].diff()


#Weather functionality, to give model some context, uses OpenMeteo API
def get_weather_data(start,end):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LAT,
        "longitude" : LON,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "hourly": "temperature_2m,weathercode",
        "timezone": "Asia/Makassar"

    }
    r = requests.get(url, params=params)
    data = r.json()
    weather_df = pd.DataFrame({
        'rounded_hour': pd.to_datetime(data['hourly']['time']),
        'temperature' : data['hourly']['temperature_2m'],
        'weathercode' : data['hourly']['weathercode']
    })
    weather_df['rounded_hour'] = weather_df['rounded_hour'].dt.floor('H')
    return weather_df


#Hours batched, to not use a lot of space and because weather happens over a period not a set time
start_time = batched_hours.min()
end_time = batched_hours.max()

print("Fetching weather data from Open-Meteo")
weather_df = get_weather_data(start_time, end_time)


#Since OpenMeteo returns weather codes and not the actual string, made a function to decode
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

#Add weather to the original dataset
weather_df['weather'] = weather_df['weathercode'].apply(decode_weather)
df = pd.merge(df, weather_df, on='rounded_hour', how='left')
df.drop(columns=['rounded_hour'], inplace=True)
print(df['weather'].value_counts(dropna=False))

#Make a new dataset, which will be processed again
df.to_csv("traffic_processed.csv",index=False)
print("✅ Data prepared and saved to traffic_processed.csv")

