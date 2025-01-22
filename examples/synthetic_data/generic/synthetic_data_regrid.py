# Use the IMMERSE_test_cases notebook to generate synthetic data (or your own)

# make sure you have installed pyic as a package using `pip install -e .` on the pyIC directory.
import xarray as xr

from pyic.grid import GRID
from pyic.regrid import make_regridder, regrid_data

grid1 = GRID("~/pyIC/examples/generate_synthetic_data/ds1.nc", ds_lon_name="lon", ds_lat_name="lat")
grid2 = GRID("~/pyIC/examples/generate_synthetic_data/ds2.nc", ds_lon_name="lon", ds_lat_name="lat")
# make regridder and use that regridder to regrid grid1.
# grid1 in the regrid_data function just needs to be a GRID#
# object with the same domain as the GRID used to make the regridder.

regridder = make_regridder(grid1, grid2, regrid_algorithm="conservative", force=True)

regridded_data = regrid_data(grid1, regridder=regridder)
regridded_data.to_netcdf("~/pyIC/examples/generate_synthetic_data/ds1_regridded.nc")
