from pyic import (
    grid,
    regrid,  # is_superset_of, subset_mask, make_regridder, regrid_data
)

# source data set
src_ds_path = "~/NCMRWF/mersea.grid_T.nc"  # make sure data is unzipped!
dst_gd_path = "~/NCMRWF/domain_cfg.nc"  # likewise

src_grid = grid.GRID(
    src_ds_path, ds_lon_name="nav_lon", ds_lat_name="nav_lat", ds_time_counter="time_counter"
)
dst_grid = grid.GRID(dst_gd_path, ds_lon_name="nav_lon", ds_lat_name="nav_lat", ds_time_counter="t")

src_data = src_grid
regrid1 = regrid.make_regridder(
    src_data,
    dst_grid,
    regrid_algorithm="bilinear",
    save_weights="~/NCMRWF/regrid_weights_bilinear.nc",
    use_inset=False,
)

regridded_T = regrid.regrid_data(src_data, regridder=regrid1)
regridded_T.to_netcdf("~/NCMRWF/regridded_mersea.grid_T_bilinear.nc")

# Let's do the same for W, but re-use the regridder.

w_data = grid.GRID(
    "~/NCMRWF/mersea.grid_W.nc", ds_lon_name="nav_lon", ds_lat_name="nav_lat", ds_time_counter="time_counter"
)
regridded_W = regrid.regrid_data(w_data, regridder=regrid1)
regridded_W.to_netcdf("~/NCMRWF/regridded_mersea.grid_W_bilinear.nc")
