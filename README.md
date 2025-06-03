# JamBuster

AI Powered Traffic Prediction using data I personally collected from my daily commute roads

## Why This Project Exists

Every day, traffic on the roads I commute on daily tends to annoy me a lot, so I thought:
> "What if I could try and fix this with the knowledge I currently have"

This project is my personal attempt at building a model that predicts traffic based on time, roads, and other engineered features

## Current Progress

- 2025-05-27: Set up the tools for collecting traffic data (API integration, custom logger script)  
- 2025-05-28: Started collecting and will continue collecting data on 3 main roads (picked Tun Abdul Razak, Boulevard, and Sultan Alauddin due to known traffic problems) — ongoing
- 2025-06-2: Started making a basic congestion predictor using Random Forests
- 2025-06-3: Finished training the basic model, first version of model will soon be ready for launch
- Model training & evaluation (coming soon)  
- Streamlit dashboard demo (planned)

## Tools and Technologies

- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Joblib
- Juypter Notebooks, Google Colab
- Streamlit (for dashboarding and UI stuff)
- Custom dataset from local roads using TomTom API

## Data

Right now, I am collecting traffic conditions mainly on the 3 key roads during different times of day. Logging:
- Time of day
- Date
- Current speed of traffic
- Expected free flow speed (average speed of a vehicle assuming 0 cars around)
- Confidence level (API integration from TomTom, indicates certainty of the data collected)

## How it works
### Data Collection:
I collected data only on roads that personally affected me. Using TomTom's API I was able to pull traffic data — things like current speed, free flow speed, and confidence levels. I then wrote a logger script that would log traffic data every 5 minutes in a loop that ran for 1000 iterations. I usually stopped logging data usually around 10-11PM since the traffic data during that timeframe wasn't really useful. Plus its when I usually call it a night anyway.
### Model Training and Model Selection:
For the early version I decided to keep it short and simple, I used a RandomForestClassifier, a tree ensemble algorithm from the Sklearn library. I would first preprocess the data adding features that I deemed viable, I also hot encoded categorical data(*ahem roads ahem*), I would also define the target y. I'd then split the preprocessed data into a training set and test set(80% for training for set, 20% for the test set - you know, the usual stuff). Lastly I would train the model on the training set, validate the prediction through the test set and record back the metrics, using F1 score. Right now RandomForestClassfier from Sklearn is all I currently need but I am eyeing XGBOOST due to it having better performance and accuracy during model training, defintely trying that algorithm that in the future.

## Changelog

- **2025-05-31**: Added the README for this project, still collecting data, moved some journal notes to this README
- **2025-06-03**: Added the model, model training notebook, logger, preprocessed data, unprocessed data, and the data preprocessor
