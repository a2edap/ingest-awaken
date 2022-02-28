import os
import cmocean
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from typing import Dict
from tsdat import DSUtil
from utils import A2ePipeline, format_time_xticks


class Pipeline(A2ePipeline):
    """--------------------------------------------------------------------------------
    AML_LD_Z01 INGESTION PIPELINE

    awaken lidar disdrometer ingest

    --------------------------------------------------------------------------------"""

    def hook_customize_raw_datasets(
        self, raw_dataset_mapping: Dict[str, xr.Dataset]
    ) -> Dict[str, xr.Dataset]:
        return raw_dataset_mapping

    def hook_customize_dataset(
        self, dataset: xr.Dataset, raw_mapping: Dict[str, xr.Dataset]
    ) -> xr.Dataset:
        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # Remove _FillValue attribute from string arrays so netcdf can save
        keys = [
            "weather_code_synop",
            "weather_code_metar",
            "weather_code_nws",
            "optics_status",
        ]
        for code in keys:
            dataset[code].attrs.pop("_FillValue")
        return dataset

    def hook_generate_and_persist_plots(self, dataset: xr.Dataset):
        return
