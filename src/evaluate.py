# src/evaluate.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


#-------------------------------- Load Data
df = pd.read_csv("../data/data.csv")

X = df[[col for col in df.columns if col != "placement_status"]]
y = df['placement_status']

_, X_test, _, y_test = train_test_split(
    X, y,
    stratify=y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE)


#------------------------------- Load Model
model = joblib.load(MODEL_PATH)


#------------------------------- Predict
y_pred = model.predict(X_test)


#------------------------------ Classification Report
print(classification_report(y_test, y_pred))



#------------------------------ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()