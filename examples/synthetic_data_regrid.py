#use the IMMERSE_test_cases notebook to generate synthetic data (or your own)
from synthetic_data_gen import make_dataset

from pyic.grid import GRID
from pyic.regrid import make_regridder, regrid_data
import xarray as xr

ds1 = make_dataset(domain='half_bowl')
ds2 = make_dataset(dx=1.,dy=1.,domain='half_bowl')
print(ds2)
ds2_crop = xr.Dataset()
for var in ds2:
    ds2_crop[var] = ds2[var][:,:,100:400,440:490]

ds1.to_netcdf('ds1.nc')
ds2_crop.to_netcdf('ds2.nc')

grid1 = GRID('ds1.nc',ds_lon_name='lon',ds_lat_name='lat')
grid2 = GRID('ds2.nc',ds_lon_name='lon',ds_lat_name='lat')

regridder = make_regridder(grid1,grid2,regrid_algorithm='conservative')

regridded_data = regrid_data(grid1,regridder = regridder)
regridded_data.to_netcdf('ds1_regridded.nc')