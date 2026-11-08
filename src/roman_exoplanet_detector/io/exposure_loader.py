"""
exposure_loader.py

Load a collection of Roman exposures.

Supports:
- *.asdf
- Roman datamodel products

Loads Exposure objects and
constructs Observation objects.

Author:
Roman Exoplanet Detector

Roman Observation Campaign
         │
         ▼
Load Multiple Exposures
         │
         ▼
Sort by Observation Time
         │
         ▼
Return Exposure Objects
"""
 

from pathlib import Path

from roman_exoplanet_detector.domain.observation import (
    Observation,
    Exposure
)

from roman_exoplanet_detector.io.roman_datamodel_loader import (
    RomanDataModelLoader
)


class ExposureLoader:

    def __init__(self, directory):

        self.directory = Path(directory)

    def files(self):

        return sorted(
            self.directory.glob("*.asdf")
        )

    def load_exposure(
        self,
        filename
    ) -> Exposure:

        loader = RomanDataModelLoader(
            filename
        )

        return loader.load()

    def load_observation(
        self,
        observation_id
    ) -> Observation:

        observation = Observation(
            observation_id
        )

        for file in self.files():

            try:

                exposure = (
                    self.load_exposure(
                        file
                    )
                )

                observation.add_exposure(
                    exposure
                )

            except Exception as exc:

                print(
                    f"Failed {file}: {exc}"
                )

        observation.sort_by_time()

        return observation

    def summary(self):

        obs = self.load_observation(
            "ROMAN_OBSERVATION"
        )

        return obs.summary()