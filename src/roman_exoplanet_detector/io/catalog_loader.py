"""
catalog_loader.py

Source catalog management.

Handles:
- Catalog I/O
- Cross-matching
- Source lookup
- Catalog merging

Roman Exposure
        │
        ▼
Source Catalog
        │
        ▼
Source Matching
        │
        ▼
Light Curve Generation
"""
from pathlib import Path
import pandas as pd

from roman_exoplanet_detector.domain.observation import (
    Source,
    SourceCatalog
)



class CatalogLoader:

    @staticmethod
    def load_csv(
        filename
    ) -> SourceCatalog:

        df = pd.read_csv(
            filename
        )

        catalog = SourceCatalog()

        for _, row in df.iterrows():

            source = Source(

                source_id=
                str(
                    row["source_id"]
                ),

                x=row["x"],

                y=row["y"],

                flux=row["flux"],

                flux_error=row.get(
                    "flux_error"
                ),

                ra=row.get("ra"),

                dec=row.get("dec")
            )

            catalog.add_source(
                source
            )

        return catalog

    @staticmethod
    def save_csv(
        catalog: SourceCatalog,
        filename
    ):

        df = catalog.to_dataframe()

        df.to_csv(
            filename,
            index=False
        )

    @staticmethod
    def merge(
        catalogs
    ) -> SourceCatalog:

        merged = SourceCatalog()

        for catalog in catalogs:

            for source in catalog.sources:

                merged.add_source(
                    source
                )

        return merged