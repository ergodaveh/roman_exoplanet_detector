"""
build_lightcurves.py
"""

from typing import Dict, List

from roman_exoplanet_detector.domain.observation import (
    Source,
    LightCurve
)


class LightCurveBuilder:

    def __init__(self):

        self.lightcurves: Dict[
            str,
            LightCurve
        ] = {}

    def add_measurement(
        self,
        source: Source,
        time: float,
        flux: float = None,
        error: float = None
    ) -> None:

        source_id = source.source_id

        if source_id not in self.lightcurves:

            self.lightcurves[source_id] = (
                LightCurve(
                    source_id=source_id
                )
            )

        lc = self.lightcurves[source_id]

        lc.add_point(

            time=time,

            flux=(
                flux
                if flux is not None
                else source.flux
            ),

            error=(
                error
                if error is not None
                else (
                    source.flux_error
                    if source.flux_error
                    else 0.0
                )
            ),
        )

    def get(
        self,
        source_id: str
    ) -> LightCurve | None:

        return self.lightcurves.get(
            source_id
        )

    def all_lightcurves(
        self
    ) -> Listreturn list(
            self.lightcurves.values()
        )

    def count(self) -> int:

        return len(
            self.lightcurves
        )