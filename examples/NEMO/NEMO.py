from pyic import (
    grid,
    regrid,  # is_superset_of, subset_mask, make_regridder, regrid_data
)

# source data set
src_dom_path = "/gws/nopw/j04/jmmp/public/pyIC/src_domain_cfg.nc"  # make sure data is unzipped!
dst_dom_path = "/gws/nopw/j04/jmmp/public/pyIC/dst_domain_cfg.nc"  # likewise
src_data_path = "/gws/nopw/j04/jmmp/public/pyIC/src_data.nc"

src_grid = grid.GRID(
    src_dom_path, ds_lon_name="nav_lon", ds_lat_name="nav_lat", ds_time_counter="time_counter"
)

src_data = grid.GRID(
    src_data_path, ds_lon_name="nav_lon", ds_lat_name="nav_lat", ds_time_counter="time_counter"
)

dst_grid = grid.GRID(dst_dom_path, ds_lon_name="glamt", ds_lat_name="gphit", ds_time_counter="time_counter")


regrid1 = regrid.make_regridder(
    src_grid,
    dst_grid,
    regrid_algorithm="bilinear",
    save_weights="~/NCMRWF/NEMO-regrid_weights_bilinear.nc",
    use_inset=False,
    parallel=False,
)

regridded_T = regrid.regrid_data(src_data, regridder=regrid1)
regridded_T.to_netcdf("/gws/nopw/j04/wcssp_india/users/jdconey/regridder_src_data_20250120.nc")
