from grid import GRID
from regrid import is_superset_of, subset_mask, regrid_data, make_regridder

if __name__ == "__main__":
    src_grid = GRID("/gws/nopw/j04/jmmp/public/pyIC/src_domain_cfg.nc", ds_lon_name="nav_lon",ds_lat_name = "nav_lat")
    dst_grid = GRID("/gws/nopw/j04/jmmp/public/pyIC/dst_domain_cfg.nc", ds_lon_name="glamt", ds_lat_name="gphit")
    regrid1 = make_regridder(src_grid,dst_grid,regrid_algorithm="conservative",save_weights="regrid_weights_conservative.nc",
                            force=True)
    src_data = GRID("/gws/nopw/j04/jmmp/public/pyIC/src_data.nc",ds_lon_name="nav_lon",ds_lat_name="nav_lat")
    subset_mask(src_data,dst_grid)
    src_data.inset = src_data.common_grid.where(src_data.lat_bool&src_data.lon_bool,drop=True)
    regridded = regrid_data(src_data,regridder=regrid1)
    regridded.to_netcdf("/gws/nopw/j04/wcssp_india/users/jdconey/regridded_data_new_con-inset.nc")