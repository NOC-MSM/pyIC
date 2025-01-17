import xarray as xr

from pyic.grid import GRID
from pyic.regrid import make_regridder, regrid_data

grid1 = GRID(
    "~/pyIC/examples/synthetic_data/generic/ds1.nc", ds_lon_name="lon", ds_lat_name="lat"
)
grid2 = GRID(
    "~/pyIC/examples/synthetic_data/1d/1D_salinity_temperature_data.nc",
    ds_lon_name="longitude",
    ds_lat_name="latitude",
    ds_time_counter="time",
)
# make regridder and use that regridder to regrid grid1.
# grid1 in the regrid_data function just needs to be a GRID#
# object with the same domain as the GRID used to make the regridder.

regridder = make_regridder(grid1, grid2, regrid_algorithm="conservative", force=False)

regridded_data = regrid_data(grid1, regridder=regridder)
regridded_data.to_netcdf("~/pyIC/examples/synthetic_data/1d/regridded_to_1d.nc")
