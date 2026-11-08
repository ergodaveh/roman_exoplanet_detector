"""
observation.py

Core data models for the Roman exoplanet pipeline.

These objects are shared across:

- roman_loader.py
- roman_datamodel_loader.py
- exposure_loader.py
- catalog_loader.py
- source_extraction.py
- build_lightcurves.py
- event_detection.py
- anomaly_detection.py
- planet_fit.py

Author:
Roman Exoplanet Detector
"""

from dataclasses import dataclass, field
from datetime import datetime

from typing import Optional
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Exposure:

    exposure_id: str

    data: np.ndarray

    err: np.ndarray | None = None

    dq: np.ndarray | None = None

    meta: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self):

        if self.data is None:

            raise ValueError(
                "Exposure data missing."
            )

    def to_dict(self):

        return {

            "exposure_id":
                self.exposure_id,

            "shape":
                list(self.data.shape),

            "meta":
                self.meta
        }


@dataclass
class Source:

    source_id: str

    x: float

    y: float

    flux: float

    flux_error: float | None = None

    ra: float | None = None

    dec: float | None = None

    metadata: dict = field(
        default_factory=dict
    )

    @property
    def snr(self):

        if (

            self.flux_error is None

            or

            self.flux_error <= 0
        ):

            return None

        return self.flux / self.flux_error

    def to_dict(self):

        return {

            "source_id":
                self.source_id,

            "x":
                self.x,

            "y":
                self.y,

            "flux":
                self.flux,

            "flux_error":
                self.flux_error,

            "ra":
                self.ra,

            "dec":
                self.dec,

            "snr":
                self.snr
        }


@dataclass
class SourceCatalog:

    sources: list[Source] = field(
        default_factory=list
    )

    def add_source(
        self,
        source: Source
    ):

        self.sources.append(source)

    def to_dataframe(self):

        return pd.DataFrame(

            [
                s.to_dict()
                for s in self.sources
            ]

        )

    @classmethod
    def from_dataframe(
        cls,
        dataframe
    ):

        catalog = cls()

        for _, row in dataframe.iterrows():

            catalog.add_source(

                Source(

                    source_id=row["source_id"],

                    x=row["x"],

                    y=row["y"],

                    flux=row["flux"],

                    flux_error=row.get(
                        "flux_error"
                    ),

                    ra=row.get("ra"),

                    dec=row.get("dec")
                )

            )

        return catalog


@dataclass
class LightCurve:

    source_id: str

    time: list[float] = field(
        default_factory=list
    )

    flux: list[float] = field(
        default_factory=list
    )

    error: list[float] = field(
        default_factory=list
    )

    def add_point(
        self,
        time,
        flux,
        error
    ):

        self.time.append(time)

        self.flux.append(flux)

        self.error.append(error)

    def dataframe(self):

        return pd.DataFrame({

            "time":
                self.time,

            "flux":
                self.flux,

            "error":
                self.error
        })


@dataclass
class Candidate:

    candidate_id: str

    source_id: str

    planet_probability: float

    anomaly_score: float

    confidence: float

    planet_mass: float | None = None

    separation: float | None = None
    
    ml_probability: float | None = None  # <-- Added missing field (maps to planet_probability or separate model output)

    current_status: str = "NEW"

    classification: str = "PLANET_CANDIDATE"

    host_mass: float | None = None  # <-- Added missing field    
  
    reviewer: str | None = None

    review_notes: str = ""

    reviews: list = field(  # <-- Added missing collection field
        default_factory=list
    )
    def review_count(self):  # <-- Added missing helper method
        return len(self.reviews)
    
    created: str = field(
        default_factory=lambda:
        datetime.utcnow().isoformat()
    )

    metadata: dict = field(
        default_factory=dict
    )

    def priority(self):

        if self.planet_probability >= 0.95:

            return "CRITICAL"

        if self.planet_probability >= 0.80:

            return "HIGH"

        if self.planet_probability >= 0.50:

            return "MEDIUM"

        return "LOW"

    def to_dict(self):

        return {

            "candidate_id":
                self.candidate_id,

            "source_id":
                self.source_id,

            "planet_probability":
                self.planet_probability,

            "anomaly_score":
                self.anomaly_score,

            "confidence":
                self.confidence,

            "priority":
                self.priority(),

            "created":
                self.created,

            "metadata":
                self.metadata
        }
    
    def summary(self):

        return self.to_dict()


@dataclass
class Observation:

    observation_id: str

    exposures: list[Exposure] = field(
        default_factory=list
    )

    metadata: dict = field(
        default_factory=dict
    )

    def add_exposure(
        self,
        exposure: Exposure
    ):

        self.exposures.append(
            exposure
        )

    def count(self):

        return len(
            self.exposures
        )