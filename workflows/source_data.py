from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import copernicusmarine
import gdown
import roms_tools as rt
import config


# -----------------------------------------
# Dataset registry (name -> handler)
# -----------------------------------------

DATASET_REGISTRY: Dict[str, Callable[["SourceData"], Path]] = {}


def register_dataset(name: str) -> Callable:
    """
    Decorator to register a dataset handler.

    Usage:
        @register_dataset("GLORYS")
        def _prepare_glorys(self): ...
    """

    def decorator(func: Callable[["SourceData"], Path]) -> Callable:
        DATASET_REGISTRY[name.upper()] = func
        return func

    return decorator


# -----------------------------------------
# SourceData
# -----------------------------------------


@dataclass
class SourceData:
    """
    Handles creation and caching of source data files (GLORYS, UNIFIED_BGC, etc.)
    for ROMS preprocessing.
    """

    grid: object
    grid_name: str
    start_time: object
    end_time: object
    datasets: List[str]  # list of dataset names (e.g., ["GLORYS", "UNIFIED_BGC"])
    clobber: bool = False

    def __post_init__(self):
        # Normalize dataset names
        self.datasets = [ds.upper() for ds in self.datasets]

        # Validate requested datasets
        known = set(DATASET_REGISTRY.keys())
        unknown = set(self.datasets) - known
        if unknown:
            raise ValueError(
                f"Unknown dataset(s) requested: {', '.join(sorted(unknown))}. "
                f"Known datasets: {', '.join(sorted(known))}"
            )

        # Per-dataset paths (generic) + convenience attrs
        self.paths: Dict[str, Path] = {}
        self.glorys_path: Path | None = None
        self.bgc_forcing_path: Path | None = None

    # -----------------------------------------
    # Public API
    # -----------------------------------------

    def prepare_all(self):
        """Prepare all requested source datasets."""
        for name in self.datasets:
            handler = DATASET_REGISTRY[name]
            path = handler(self)  # call handler with this instance
            self.paths[name] = path  # store generically

        return self

    # -----------------------------------------
    # Internals / helpers
    # -----------------------------------------

    def _construct_glorys_path(self) -> Path:
        fn = (
            f"GLORYS_{self.grid_name}_"
            f"{self.start_time.strftime('%Y-%m-%d')}-"
            f"{self.end_time.strftime('%Y-%m-%d')}.nc"
        )
        return config.paths.source_data / fn

    # ---------------------------
    # GLORYS handler
    # ---------------------------

    @register_dataset("GLORYS")
    def _prepare_glorys(self) -> Path:

        path = self._construct_glorys_path()
        needs_download = self.clobber or (not path.exists())

        if needs_download:
            if path.exists():
                print(f"⚠️  Clobber=True: removing existing GLORYS file {path.name}")
                path.unlink()

            print(f"⬇️  Downloading GLORYS → {path}")
            copernicusmarine.subset(
                dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
                variables=["thetao", "so", "uo", "vo", "zos"],
                **rt.get_glorys_bounds(self.grid),
                start_datetime=self.start_time,
                end_datetime=self.end_time,
                coordinates_selection_method="outside",
                output_filename=path.name,
                output_directory=config.paths.source_data,
            )
        else:
            print(f"✔️  Using existing GLORYS file: {path}")

        self.glorys_path = path
        return path

    # ---------------------------
    # UNIFIED BGC handler
    # ---------------------------

    @register_dataset("UNIFIED_BGC")
    def _prepare_unified_bgc_dataset(self) -> Path:
        url_bgc_forcing = (
            "https://drive.google.com/uc?id=1wUNwVeJsd6yM7o-5kCx-vM3wGwlnGSiq"
        )
        path = config.paths.source_data / "BGCdataset.nc"
        needs_download = self.clobber or (not path.exists())

        if needs_download:
            if path.exists():
                print(f"⚠️  Clobber=True: removing existing BGC file {path.name}")
                path.unlink()

            print(f"⬇️  Downloading BGC dataset → {path}")
            gdown.download(url_bgc_forcing, str(path), quiet=False)
        else:
            print(f"✔️  Using existing BGC dataset: {path}")

        self.bgc_forcing_path = path
        return path
