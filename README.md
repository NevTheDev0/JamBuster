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

## Installation/Running Models or Scripts locally

*Note: This is for the whole pipeline*
### 1. Clone the Repo
```
git clone https://github.com/NevTheDev0/JamBuster
cd JamBuster
```
### 2. Install requirement
```
pip install -r requirements.txt
```
### 3. Set Up API Key
*Note: The API we will be using is the TomTom Traffic API (https://developer.tomtom.com/traffic-api/documentation/product-information/introduction)*

Create a .env file in the root directory and add your key like this:
```
TRAFFIC_API_KEY=your_api_key_here
```
**WARNING: NEVER SHARE YOUR API KEY!**

### 4. Run the Traffic Logger
You can either run it in your IDE of choice(I personally use VScode) or by running it in your terminal:
```
scripts/traffic_logger.py
```

### 5. Train the Model
Again you can run this script directly in your IDE or run it in your terminal:
```
scripts/Model_Train.py
```

### 6. Make Predictions
```
scripts/PredictionPles.py
```
---
#### Note this is for V1 of the model using Random Trees(Sklearn) as of 2025-06-05 I have not yet uploaded some scripts will update these later(Its like 2 am rn LOLLL)


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
For the early version I decided to keep it short and simple, I used a RandomForestClassifier, a tree ensemble algorithm from the Sklearn library. I would first preprocess the data adding features that I deemed viable, I also hot encoded categorical data(*ahem roads ahem*), I would then also define the target y. I'd then split the preprocessed data into a training set and test set(80% for training for set, 20% for the test set - you know, the usual stuff). Lastly I would train the model on the training set, validate the prediction through the test set and record back the metrics, using F1 score. Right now RandomForestClassfier from Sklearn is all I currently need but I am eyeing XGBOOST due to it having better performance and accuracy during model training, defintely trying that algorithm that in the future.

## Changelog

- **2025-05-31**: Added the README for this project, still collecting data, moved some journal notes to this README
- **2025-06-03**: Added, model training notebook, preprocessed data, unprocessed data, and the data preprocessor
- **2025-06-04**: Forgot to commit the folders and files above, actually commited them this time(My bad lol)
- **2025-06-05**: Updated this README to include how to run the model locally, also added traffic_logger.py, will upload more necessary files soon
