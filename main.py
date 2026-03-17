from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("student-mat.csv", sep=";")

df["Result"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
df = df.drop(["G1", "G3"], axis=1)

X = df.drop("Result", axis=1)
y = df["Result"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ✅ Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("Model & files saved successfully ✅")
