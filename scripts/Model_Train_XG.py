from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import joblib



X,y = joblib.load("models/train_ready_data.pkl")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)



sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

xgb = XGBClassifier(
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric = 'logloss',
    verbosity=0 
)

xgb.fit(X_train_resampled, y_train_resampled)

y_pred = xgb.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

scores = cross_val_score(xgb, X, y, cv=5, scoring='f1')
print("🔥 CV F1 Score (XGBoost):", scores.mean())

joblib.dump(xgb,"XGBClassifier.pkl")