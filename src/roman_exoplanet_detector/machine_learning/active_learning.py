"""
active_learning.py

Human-in-the-loop learning.
"""

import pandas as pd
import numpy as np


class ActiveLearningManager:

    def __init__(self):

        self.training_set = (
            pd.DataFrame()
        )

    def status_to_label(
        self,
        status
    ):

        mapping = {

            "Accepted": 1,

            "Rejected": 0,

            "Possible Binary Lens": 0,

            "Needs Further Study": -1
        }

        return mapping.get(
            status,
            -1
        )

    def add_reviewed_candidate(

        self,

        features,

        review_status

    ):

        row = (
            dict(features)
        )

        row["label"] = (

            self.status_to_label(
                review_status
            )
        )

        self.training_set = pd.concat(

            [
                self.training_set,
                pd.DataFrame([row])
            ],

            ignore_index=True
        )

    def uncertainty_score(
        self,
        probability
    ):
        """
        Maximum uncertainty at 0.5
        """

        return (
            1
            -
            abs(
                probability
                -
                0.5
            )
            * 2
        )

    def select_candidates_for_review(

        self,

        candidate_table,

        top_n=100

    ):

        candidate_table = (
            candidate_table.copy()
        )

        candidate_table[
            "uncertainty"
        ] = (

            candidate_table[
                "planet_probability"
            ].apply(

                self.uncertainty_score
            )
        )

        return (

            candidate_table

            .sort_values(
                "uncertainty",
                ascending=False
            )

            .head(top_n)
        )

    def reviewer_disagreement(

        self,

        review_labels

    ):

        unique = set(
            review_labels
        )

        if len(unique) <= 1:

            return 0

        return (
            len(unique) - 1
        ) / len(unique)

    def retraining_needed(

        self,

        minimum_samples=1000

    ):

        labeled = (

            self.training_set[
                self.training_set[
                    "label"
                ] >= 0
            ]
        )

        return (

            len(labeled)

            >=

            minimum_samples
        )

    def training_summary(
        self
    ):

        labeled = (
            self.training_set[
                self.training_set[
                    "label"
                ] >= 0
            ]
        )

        return {

            "total":
                len(
                    self.training_set
                ),

            "labeled":
                len(
                    labeled
                ),

            "accepted":

                int(
                    (
                        labeled[
                            "label"
                        ] == 1
                    ).sum()
                ),

            "rejected":

                int(
                    (
                        labeled[
                            "label"
                        ] == 0
                    ).sum()
                )
        }