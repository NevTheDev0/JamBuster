# JamBuster

AI Powered Traffic Prediction using data I personally collected from my daily commute roads

## Why This Project Exists

Every day, traffic on the roads I commute on daily tends to annoy me a lot, so I thought:
> "What if I could try and fix this with the knowledge I currently have"

This project is my personal attempt at building a model that predicts traffic based on time, roads, and other engineered features

## Current Progress

- 2025-05-27: Set up the tools for collecting traffic data (API integration, custom logger script)  
- 2025-05-28: Started collecting and will continue collecting data on 3 main roads (picked Tun Abdul Razak, Boulevard, and Sultan Alauddin due to known traffic problems) — ongoing  
- Model training & evaluation (coming soon)  
- Streamlit dashboard demo (planned)

## Tools and Technologies

- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit (for dashboarding and UI stuff)
- Custom dataset from local roads using TomTom API

## Data

Right now, I am collecting traffic conditions mainly on the 3 key roads during different times of day. Logging:
- Time of day
- Date
- Current speed of traffic
- Expected free flow speed (average speed of a vehicle assuming 0 cars around)
- Confidence level (API integration from TomTom, indicates certainty of the data collected)

## Changelog

- **2025-05-31**: Added the README for this project, still collecting data, moved some journal notes to this README
