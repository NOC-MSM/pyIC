import numpy as np
import pytest
import xarray as xr

from pyic.grid import GRID
from pyic.regrid import make_regridder, regrid_data

from .synthetic_data_gen import main as synmain

synmain()


@pytest.mark.parametrize(("filepath"), (["ds1.nc", "ds2.nc", "ds1_64.nc"]))
def test_grid_ds(filepath):
    g1 = GRID(data_filename=filepath, ds_lon_name="lon", ds_lat_name="lat")
    ds = xr.open_dataset(filepath)
    xr.testing.assert_identical(g1.ds, ds)


def test_synthetic_regrid():
    g1 = GRID(data_filename="ds1_crop.nc", ds_lon_name="lon", ds_lat_name="lat")
    g2 = GRID(data_filename="ds2.nc", ds_lon_name="lon", ds_lat_name="lat")
    regridder = make_regridder(g2, g1, check_superset=False)
    g2_regrid = regrid_data(g2, regridder=regridder)
    ds = xr.open_dataset("ds2_rgd.nc").isel(time_counter=0)
    np.testing.assert_array_equal(g2_regrid["temperature"].values, ds["temperature"].values)


def test_synthetic_vertical_regrid():
    g1 = GRID(data_filename="ds1_32.nc", ds_lon_name="lon", ds_lat_name="lat")
    g2 = GRID(data_filename="ds2.nc", ds_lon_name="lon", ds_lat_name="lat")
    regridder = make_regridder(g1, g2, check_superset=False)
    g1_regrid = regrid_data(
        g1,
        regridder=regridder,
        regrid_vertically=True,
        vertical_kwargs={"levels": np.arange(0, 3600, 100), "vertical_coord": "depth", "method": "linear"},
    )
    np.testing.assert_array_equal(g1_regrid["temperature"].shape, g2.common_grid["temperature"].shape)
