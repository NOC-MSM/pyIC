from pyic.grid import GRID
from pyic.regrid import make_regridder, regrid_data
import xarray as xr

grid1 = GRID('/home/users/jdconey/pyIC/examples/generate_synthetic_data/ds1.nc',ds_lon_name = 'lon',ds_lat_name = 'lat')
grid2 = GRID('/home/users/jdconey/pyIC/examples/generate_synthetic_data/1D_salinity_temperature_data_2.nc', ds_lon_name = 'longitude', ds_lat_name = 'latitude', ds_time_counter='time')
#make regridder and use that regridder to regrid grid1.
#grid1 in the regrid_data function just needs to be a GRID#
#object with the same domain as the GRID used to make the regridder.

regridder = make_regridder(grid1,grid2,regrid_algorithm='conservative')

regridded_data = regrid_data(grid1,dest_grid=grid2)#regridder = regridder)
regridded_data.to_netcdf('/home/users/jdconey/pyIC/examples/generate_synthetic_data/1d_data_regridded.nc')