"""
Visualization tools.
"""

import matplotlib.pyplot as plt


def plot_image(image):

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        image,
        origin="lower",
        cmap="gray"
    )

    plt.colorbar()

    plt.title(
        "Roman Image"
    )

    plt.show()


def plot_lightcurve(
        time,
        flux):

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time,
        flux,
        "."
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Flux"
    )

    plt.title(
        "Light Curve"
    )

    plt.show()


def plot_fit(
        time,
        flux,
        model):

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time,
        flux,
        ".",
        label="Observed"
    )

    plt.plot(
        time,
        model,
        color="red",
        label="Model"
    )

    plt.legend()

    plt.show()


def plot_residuals(
        time,
        residuals):

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        time,
        residuals
    )

    plt.axhline(
        0,
        color="red",
        linestyle="--"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Planet Anomaly Search"
    )

    plt.show()