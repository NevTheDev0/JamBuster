from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib


X, y = joblib.load("models/train_ready_data.pkl")

print(len(X), len(y))


#Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

print("Before:", y_train.value_counts(normalize=True))
print("After :", y_train_resampled.value_counts(normalize=True))

#Classifier training
classifier = RandomForestClassifier(n_estimators=100, random_state=42,class_weight="balanced")
classifier.fit(X_train_resampled, y_train_resampled)

#Predict
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
scores = cross_val_score(classifier, X, y, cv=5, scoring='f1')
print("🔥 CV F1 Score:", scores.mean())


joblib.dump(classifier, "models/RFclassifier.pkl")




