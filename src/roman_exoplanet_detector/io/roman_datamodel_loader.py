"""
roman_datamodel_loader.py

Mission-aware Roman loader.

Returns typed Exposure objects.

Observation
Exposure
Source
SourceCatalog
LightCurve
Candidate
"""

from pathlib import Path

from roman_exoplanet_detector.domain.observation import (
    Exposure
)

try:
    from roman_datamodels import open as roman_open
except ImportError:
    roman_open = None


class RomanDataModelLoader:

    def __init__(self, filename):

        self.filename = Path(filename)

    def _metadata_to_dict(self, meta):

        try:
            return meta.to_flat_dict()
        except Exception:
            try:
                return dict(meta)
            except Exception:
                return {}

    def load(self):

        if roman_open is None:

            raise ImportError(
                "roman_datamodels not installed."
            )

        model = roman_open(
            str(self.filename)
        )

        exposure = Exposure(

            exposure_id=
            self.filename.stem,

            data=
            model.data,

            err=
            getattr(model, "err", None),

            dq=
            getattr(model, "dq", None),

            meta=
            self._metadata_to_dict(
                model.meta
            )
        )

        return exposure

    def summary(self):

        exposure = self.load()

        return exposure.summary()