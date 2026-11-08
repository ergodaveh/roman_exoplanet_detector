"""
reviewer_tools.py
"""

import numpy as np


def novelty_score(
    planet_mass,
    separation
):

    score = 0

    if planet_mass is not None:

        if planet_mass < 2:
            score += 40

        elif planet_mass < 10:
            score += 20

    if separation is not None:

        if separation > 5:
            score += 30

        elif separation > 2:
            score += 15

    return score


def uncertainty_score(errors):

    if len(errors) == 0:
        return 0

    return (
        np.mean(errors)
        * 100
    )


def anomaly_priority(
    anomaly_score
):

    return anomaly_score * 5


def confidence_modifier(
    confidence
):

    return (
        confidence
        * 20
    )


def review_score(
    planet_mass,
    separation,
    anomaly_score,
    parameter_errors,
    confidence
):

    score = (

        novelty_score(
            planet_mass,
            separation
        )

        +

        anomaly_priority(
            anomaly_score
        )

        +

        uncertainty_score(
            parameter_errors
        )

        +

        confidence_modifier(
            confidence
        )
    )

    return round(
        score,
        2
    )


def recommendation(score):

    if score >= 100:
        return "CRITICAL"

    if score >= 80:
        return "HIGH"

    if score >= 50:
        return "MEDIUM"

    return "LOW"


def reviewer_agreement(
    review_statuses
):

    unique = set(
        review_statuses
    )

    if len(unique) == 1:
        return 1.0

    return (
        1.0
        /
        len(unique)
    )