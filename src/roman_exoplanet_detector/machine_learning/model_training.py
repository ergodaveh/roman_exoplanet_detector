"""
model_training.py

Research-grade classifier training.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.calibration import (
    CalibratedClassifierCV
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_recall_curve
)


class CandidateClassifier:

    def __init__(self):

        base_model = (
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                class_weight="balanced"
            )
        )

        self.model = (
            CalibratedClassifierCV(
                estimator=base_model,
                cv=5,
                method="isotonic"
            )
        )

    def train(
        self,
        X,
        y
    ):

        self.model.fit(
            X,
            y
        )

    def cross_validate(
        self,
        X,
        y
    ):

        folds = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        auc_scores = cross_val_score(

            self.model,

            X,

            y,

            scoring="roc_auc",

            cv=folds
        )

        return {

            "mean_auc":
                np.mean(
                    auc_scores
                ),

            "std_auc":
                np.std(
                    auc_scores
                ),

            "all_scores":
                auc_scores
        }

    def evaluate(
        self,
        X_test,
        y_test
    ):

        probabilities = (
            self.model.predict_proba(
                X_test
            )[:, 1]
        )

        predictions = (
            self.model.predict(
                X_test
            )
        )

        precision, recall, _ = (

            precision_recall_curve(

                y_test,

                probabilities
            )
        )

        return {

            "roc_auc":
                roc_auc_score(
                    y_test,
                    probabilities
                ),

            "classification_report":
                classification_report(
                    y_test,
                    predictions
                ),

            "precision":
                precision,

            "recall":
                recall
        }

    def predict(
        self,
        X
    ):

        return {

            "planet_probability":
                self.model.predict_proba(
                    X
                )[:, 1],

            "prediction":
                self.model.predict(
                    X
                )
        }

    def save(
        self,
        filename
    ):

        joblib.dump(
            self.model,
            filename
        )

    def load(
        self,
        filename
    ):

        self.model = (
            joblib.load(
                filename
            )
        )