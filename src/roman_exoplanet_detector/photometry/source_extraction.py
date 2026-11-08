"""
source_extraction.py
"""

import numpy as np

from photutils.background import (
    Background2D,
    MedianBackground
)

from photutils.segmentation import (
    detect_threshold,
    detect_sources,
    SourceCatalog
)


class SourceExtractor:

    def __init__(
        self,
        image
    ):
        self.image = image

    def remove_background(self):

        bkg = Background2D(
            self.image,
            box_size=(64, 64),
            filter_size=(3, 3),
            bkg_estimator=MedianBackground()
        )

        return (
            self.image
            -
            bkg.background
        )

    def extract_catalog(
        self,
        nsigma=2.0,
        npixels=10
    ):

        image = (
            self.remove_background()
        )

        print(
            "Image statistics:",
            image.min(),
            image.max(),
            image.mean(),
            image.std()
        )

        threshold = detect_threshold(
            image,
            n_sigma=nsigma
        )
        print(
            "Threshold statistics:",
            threshold.min(),
            threshold.max()
        )
        segments = detect_sources(
            image,
            threshold,
            n_pixels=npixels
        )
        print(
            "Segments:",
            segments
        )
        if segments is None:

            print(
                "No sources detected."
            )

            return []

        catalog = SourceCatalog(
            image,
            segments
        )

        return catalog.to_table()