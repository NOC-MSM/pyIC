# barebones example similar to pyic_exe for India regridding.

from pyic.grid import GRID
from pyic.regrid import make_regridder, regrid_data

grid1 = GRID(
    data_filename="mesh_mask_ORCA025_light.nc4",
    ds_lon_name="glamt",
    ds_lat_name="gphit",
    ds_time_counter="time",
)
grid2 = GRID(
    data_filename="domain_cfg.nc",
    ds_time_counter="t",
    ds_lat_name="gphit",
    ds_lon_name="glamt",
    ds_z_name="z",
)
regridder = make_regridder(grid1, grid2, save_weights="weights.nc", landsea_mask="tmask")
grid1_regrid = regrid_data("mersea.grid_T.nc", regridder=regridder)
grid1_regrid.to_netcdf("regridded_20250603.nc")
