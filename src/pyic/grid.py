import numpy as np
import xarray as xr
import xesmf as xe


class GRID:
    """Group of methods to encode gridded data set to enable regridding for NEMO."""

    def open_dataset(self, filename):
        return xr.open_dataset(filename)

    def get_dim_varname(self, dimtype):
        if dimtype == "longitude":
            matches = self.lon_names
        elif dimtype == "latitude":
            matches = self.lat_names
        for var in matches:
            if var in self.ds:
                return var
        raise Exception(
            f"Missing variable name for {dimtype} in data set. I tried these {matches} already."
         +f"Specify the variable name for {dimtype} explicitly using the 'ds_{dimtype[:3]}_name' argument."
        )

    def extract_lonlat(self, lon_name=None, lat_name=None):
        if lon_name is not None:
            if lon_name in self.ds:
                lon_da = self.ds[lon_name]
            else:
                raise Exception(f"{lon_name} not in given data set.")
        else:
            lon_da = self.ds[self.get_dim_varname("longitude")]
        if lat_name is not None:
            if lat_name in self.ds:
                lat_da = self.ds[lat_name]
            else:
                raise Exception(f"{lat_name} not in given data set.")
        else:
            lat_da = self.ds[self.get_dim_varname("latitude")]

        if len(lon_da.shape) == 1:
            lon_da = xr.DataArray(np.meshgrid(lon_da, lon_da), dims=["y", "x"])
        if len(lat_da.shape) == 1:
            lat_da = xr.DataArray(np.meshgrid(lat_da, lat_da), dims=["y", "x"])
        return lon_da, lat_da

    def make_common_grid(self, lon_name, lat_name):
        ds_grid = self.ds.isel(time_counter=0).rename(
            {lon_name: "lon", lat_name: "lat"}
        )
        ds_grid.set_coords(("lat", "lon"))
        return ds_grid

    def __init__(self, data_filename=None, ds_lon_name=None, ds_lat_name=None):
        """Initialise the class.

        data_filename: path to a data set on the desired grid.
        ds_lon_name: optional, the name in the data set of the longitude variable.
                     If none, will be inferred from common ones.
        ds_lat_name: optional, the name in the data set of the latitude variable.
                     If none, will be inferred from common ones.
        """
        self.data_filename = data_filename
        self.lon_names = ["nav_lon"]
        self.lat_names = ["nav_lat"]
        self.ds = self.open_dataset(self.data_filename)
        self.lon, self.lat = self.extract_lonlat(ds_lon_name, ds_lat_name)
        self.common_grid = self.make_common_grid(ds_lon_name, ds_lat_name)
        self.coords = {"lon_name": ds_lon_name, "lat_name": ds_lat_name}
        # return self.ds,self.lat,self.lon

    def regrid(
        self,
        destination_grid,
        regrid_algorithm="bilinear",
        save_weights=None,
        reload_weights=None,
    ):
        """Regrid the source grid onto a new grid.

        destination_grid: instance of GRID class (see grid.py),
            containing GRID of the destination grid (i.e. NEMO grid).

        regrid_algorithm: optional, str, should be one of
            ["bilinear", "conservative", "conservative_normed", "patch", "nearest_s2d", "nearest_d2s"],
            passed to xesmf.Regridder, see xesmf documentation for details.
        save_weights: optional str, if want to save regridding_weights then should be
                        "path/to/weights.nc", otherwise ignored.
        reload_weights: optional str, if want to load regridding_weights from a file then should be
                        "path/to/weights.nc", otherwise ignored and weights will be calculated by xesmf.
        """
        regridder = xe.Regridder(
            ds_in=self,
            ds_out=destination_grid,
            method=regrid_algorithm,
            periodic=True,
            ignore_degenerate=True,
            unmapped_to_nan=True,
            weights=reload_weights,
        )
        if save_weights is not None:
            regridder.to_netcdf(save_weights)
        return regridder


if __name__ == "__main__":
    src_grid = GRID("/projectsa/NEMO/joncon/pyIC_data/src_domain_cfg.nc")
    dest_grid = GRID("/projectsa/NEMO/joncon/pyIC_data/dest_domain_cfg.nc")
