# src/train.py

import joblib
import pandas as pd

from config import (
    TARGET,
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_PATH
)

from preprocessing import build_preprocessor

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score)


#------------------------- Load Data


df = pd.read_csv('/Users/festusattornelson/Documents/Projects/Python_Udemy/Projects/StudentPerformance/data/raw/student_dataset_10000_rows.csv')

X = df[[col for col in df.columns if col != "placement_status"]]
y = df['placement_status']


#---------------------- Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    stratify=y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


#--------------------- Preprocessor
preprocessor = build_preprocessor(X_train)



#--------------------- Models
models = {

    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",
            LogisticRegression(max_iter=2000))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",
            RandomForestClassifier(random_state=RANDOM_STATE))
    ]),

    "Gradient Boosting": Pipeline([
        ("preprocessor", preprocessor),
        ( "classifier",
            GradientBoostingClassifier(
                random_state=RANDOM_STATE))
    ])
}



#------------------------ Hyperparameters
param_grids = {

    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10]},

    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20]},

    "Gradient Boosting": {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.01, 0.1]}
}


#-------------------------- CV Setup
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted"
}

results = []

best_model = None
best_score = -1


#-------------------------- Train Loop
for name, pipeline in models.items():

    print(f"\nTraining {name}")

    grid = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )

    y_pred = model.predict(X_test)

    result = {
        "Model": name,

        "CV Accuracy":
            cv_scores["test_accuracy"].mean(),

        "CV Precision":
            cv_scores["test_precision"].mean(),

        "CV Recall":
            cv_scores["test_recall"].mean(),

        "CV F1":
            cv_scores["test_f1"].mean(),

        "Test Accuracy":
            accuracy_score(y_test, y_pred),

        "Test Precision":
            precision_score(
                y_test,
                y_pred,
                average="weighted"
            ),

        "Test Recall":
            recall_score(
                y_test,
                y_pred,
                average="weighted"
            ),

        "Test F1":
            f1_score(
                y_test,
                y_pred,
                average="weighted"
            ),

        "Best Params":
            grid.best_params_
    }

    results.append(result)

    if result["Test F1"] > best_score:

        best_score = result["Test F1"]
        best_model = model


#-------------------------- Save Results
results_df = pd.DataFrame(results)

print("\nResults")
print(results_df)

joblib.dump(
    best_model,
    MODEL_PATH
)

print(f"\nBest model saved to {MODEL_PATH}")