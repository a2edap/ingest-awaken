import os
import xarray as xr
from utils import expand, set_dev_env
from ingest.wind_cube_nwtc import Pipeline

parent = os.path.dirname(__file__)


# TODO – Developer: Update paths to your input files here.
def test_pipeline_at_nwtc():
    set_dev_env()
    pipeline = Pipeline(
        expand("config/pipeline_config_nwtc.yml", parent),
        expand("config/storage_config.yml", parent),
    )
    output = pipeline.run(expand("tests/data/input/data.csv", parent))
    expected = xr.open_dataset(expand("tests/data/expected/data.csv", parent))
    xr.testing.assert_allclose(output, expected)
