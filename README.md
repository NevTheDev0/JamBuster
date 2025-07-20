# JamBuster

AI-Powered Traffic Prediction Using Data from My Daily Commute

## Why This Project Exists

Every day, traffic on the roads I commute on tends to drive me nuts, so I thought:
> "What if I could fix this using the skills I’ve built so far?"

This project is my personal attempt at building a model that predicts traffic based on time, roads, and other engineered features

## Current Progress

- 2025-05-27: Set up tools for collecting traffic data (API integration, custom logger script)  
- 2025-05-28: Began collecting and will continue collecting data on 3 main roads (Tun Abdul Razak, Boulevard, and Sultan Alauddin due to known traffic problems) — ongoing
- 2025-06-02: Started making a basic congestion predictor using Random Forest
- 2025-06-03: Finished training the basic model, first version of the model almost ready for launch
- 2025-06-04: Trained first version of the model on Random Forest
- 2025-06-05: Model wasn't generalizing, reworked to improve generalization
- 2025-06-06: Model trained and evaluated on RME, CV, F1 Score
- 2025-06-07: Simulated Prediction (locally, not live), results were underwhelming
- 2025-06-08: Split preprocessing stage into two parts: preparing the data then building the pipeline
- 2025-06-09: Integrated OpenMeteo for weather during data preprocessing 
- 2025-06-10: Retrained model on preprocessed data, re-evaluated RME, CV, and F1 Score
- 2025-06-11: First version of real time simulation was created
- 2025-06-12: Fixed critical issue in transformers.py specifically in FeatureDropperTransformers
- 2025-06-13: Fine-tuned model
- 2025-06-14: Polished real-time simulation script, now working as intended
- 2025-06-15: Began training new model on XGBoost, working to improve this model
- 2025-06-16: XGBoost doesn't seem to be doing as well as Random Forest, attempting to achieve similar results
- 2025-06-17: Fine-tuned XGBoost to find the best outcome 
- 2025-06-20: Uploaded almost every component into this GitHub repo
- 2025-06-21: Reworked the simulation script to use XGBoost instead of RandomForest, decision based on performance
- 2025-06-25: Streamlit dashboard demo (alpha version done, needs fine tuning and polish)
- 2025-07-20: Realized dataset was horribly imbalanced and skewed, refining traffic log code to only log dirty data to help balance this out

## Tools and Technologies

- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Joblib
- Jupyter Notebooks, Google Colab
- Streamlit (for dashboarding and UI)
- Custom dataset from local roads using TomTom API

## Installation Guides – V1 (This version simulates live predictions using local data)

## SECTION 1: This section will be a guide on how to reproduce the model locally

### 1. Clone the Repo
```
git clone https://github.com/NevTheDev0/JamBuster
cd JamBuster
```
⚠️ Note on Dataset
Due to licensing restrictions, the raw traffic dataset (`traffic_log.csv`) has been removed from this repository.
However, you can regenerate it by:
1. Getting your own [TomTom API Key](https://developer.tomtom.com/).
2. Running `scripts/traffic_logger.py`, which collects and appends real-time traffic data into `traffic_log.csv`.
See `scripts/traffic_logger.py` for implementation.

### 2. Install requirements
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
python scripts/traffic_logger.py
```
*Note: I recommend running this in a separate terminal. You could try multithreading... but that headache’s on you.*

### 5. Preprocess the Data
Make sure you already have a few data points in traffic_log.csv(assuming you ran traffic_logger.py) before running the preprocessing script, or it'll throw you an error:

```
python scripts/prep_data.py
```

### 6. Build the Pipeline(for further processing)
You will need `transformers.py`, which I have provided in the `scripts` folder
```
python scripts/build_pipeline.py
```
*Note: This will output a pkl file, if you ran it correctly it will simply update the pkl files in the "models" folder*

### 7. Train the Model
Make sure you follow each step in chronological order, for the training below you'll definitely need the pipeline, you can run this script directly in your IDE or run it in your terminal:
```
python scripts/Model_Train_RF.py #This is for the Random Forest version
```
OR 
```
python scripts/Model_Train_XG.py #This is for the XGBoost version
```

### 8. Make Predictions
```
python scripts/sim_RT_prediction.py
```
---
## SECTION 2: This section will be a guide on how to use the model purely for predictions
### 1. Clone the Repo
```
git clone https://github.com/NevTheDev0/JamBuster
cd JamBuster
```

### 2. Install requirements
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
python scripts/traffic_logger.py
```
*Note: again as mentioned previously above in section 1, you may multithread OR go the simple route and use another terminal*

### 5. Make Predictions
```
python scripts/sim_RT_prediction.py
```

### ALTERNATIVE STEP
Since some of you may prefer to run the prediction through a simple CLI file, I have provided that too. All you need to do is install the requirements(Step 2), then to run predictions through the CLI script simply copy and paste(or type, I won't judge) the line below into your preferred terminal:
```
python JamBusterCLI.py run
```

---
> V1, as of the 23rd of June 2025, is currently using XGBoost as the base model when making predictions in the simulation



## Data

Right now, I am collecting traffic conditions mainly on the 3 key roads during different times of day. Logging:
- Time of day
- Date
- Current speed of traffic
- Expected free flow speed (average speed of a vehicle assuming 0 cars around)
- Confidence level (API integration from TomTom, indicates certainty of the data collected)

## How it works
### Data Collection:
I collected data only on roads that personally affected me. Using TomTom's API I was able to pull traffic data — such as current speed, free flow speed, and confidence levels. I then wrote a logger script that would log traffic data every 5 minutes in a loop that ran for 1000 iterations. I usually stopped logging data around 10-11PM since the traffic data during that timeframe wasn't really useful, which is also when I call it a day.
### Model Training and Model Selection:
For the early version I decided to keep it short and simple, I used a RandomForestClassifier, a tree ensemble algorithm from the Sklearn library. I first preprocessed the data, added features that I deemed viable, and one-hot encoded categorical data(*ahem roads ahem*), I'd then define the target variable y. I'd then split the preprocessed data into a training set and test set(80% for training set and 20% for the test set — you know, the usual stuff). Lastly I would train the model on the training set, validate the prediction through the test set and record back the metrics, using F1 score.

While the Random Forest version was great, I also tried XGBoost, a gradient boosted decision tree algorithm. I followed the same preprocessing steps I used for Random Forest; the only difference was the classifier training, which used XGBoost instead. The model was pretty janky at first, but a few hyperparameter tweaks got it working as intended. Results were almost head to head, but one true deciding factor was how well the model did on unseen data, since I am hoping to predict future traffic jams, not ones that already happened.

After a quick thought, deciding to compare each model on their metrics, I booted up a new Jupyter Notebook file to do so. Here I was hoping to be able to visualize how each model did, I compared their precision, their recall, F1 score, and Cross Validation. These comparisons gave me insight on how they worked, while Random Forest performed very well on the data we gave it, XGBoost demonstrated good performance on data it has never seen before. Ultimately, I chose XGBoost because it aligned more with what I was trying to achieve here.
*Note: If you would like to see my thought process on model comparisons refer to `notebooks/03_Model_Comparison.ipynb`*

## Changelog
- **2025-05-31**: Added the README for this project, still collecting data, moved some journal notes to this README
- **2025-06-03**: Added, model training notebook, preprocessed data, unprocessed data, and the data preprocessor
- **2025-06-04**: Forgot to commit the folders and files above, actually committed them this time(I fell asleep, I'm human too okay)
- **2025-06-05**: Updated this README to include how to run the model locally, also added traffic_logger.py, will upload more necessary files soon
- **2025-06-17**: Fine tuned models and evaluated their RME, CV, and F1 Score, will upload soon
- **2025-06-20**: Fine tuned models further before uploading, successfully uploaded scripts folder and models folder, will update notebooks and will include research report
- **2025-06-21**: Changed sim_RT_predictions.py to use XGBoost instead of Random Forest
- **2025-06-22**: Removed a folder containing sensitive information
- **2025-07-20**: Visualized dataset, suprisingly skewed towards class 0s, class 1s are very rare and never meet the threshold, updating the code and will retrain model on dirty dataset with high congestion
