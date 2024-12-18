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
            + f"Specify the variable name for {dimtype} explicitly using the 'ds_{dimtype[:3]}_name' argument."
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

    def make_common_coords(self, lon_name, lat_name):
        """Put grid onto grid with lon and lat for coordinate names ready for regridding.

        lon_name: given lon coordinate name
        lat_name: given lat coordinate name
        """
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
        self.lon_names = ["glamt,","x"]
        self.lat_names = ["gphit","y"]
        self.ds = self.open_dataset(self.data_filename)
        self.lon, self.lat = self.extract_lonlat(ds_lon_name, ds_lat_name)
        self.common_grid = self.make_common_coords(ds_lon_name, ds_lat_name)
        self.coords = {"lon_name": ds_lon_name, "lat_name": ds_lat_name}
        # return self.ds,self.lat,self.lon

    def is_superset_of(self,destination_grid,return_indices=True):
        """Check if source grid is a superset of the destination grid
        """
        if self.common_grid['lat'].min()<=destination_grid.common_grid['lat'].min() or destination_grid.common_grid['lat'].min()<-89:
            if self.common_grid['lat'].max()>=destination_grid.common_grid['lat'].max() or destination_grid.common_grid['lat'].max()>89:
                if self.common_grid['lon'].min()<=destination_grid.common_grid['lon'].min() or destination_grid.common_grid['lon'].min()<-179:
                    if self.common_grid['lon'].max()>=destination_grid.common_grid['lon'].max() or destination_grid.common_grid['lon'].max()>179:
                        print('Source grid is a superset of destination grid.')
                        if return_indices:
                            return self.make_subset(destination_grid)
                    else:
                        raise Exception(f"Source not superset of destination: max longitude. {self.common_grid['lon'].max().values} < {destination_grid.common_grid['lon'].max().values}.")
                else:
                    raise Exception(f"Source not superset of destination: min longitude. {self.common_grid['lon'].min().values} > {destination_grid.common_grid['lon'].min().values}.")
            else:
                raise Exception(f"Source not superset of destination: max latitude. {self.common_grid['lat'].max().values} < {destination_grid.common_grid['lat'].max().values}.")
        else:
            raise Exception(f"Source not superset of destination: min latitude. {self.common_grid['lat'].min().values} > {destination_grid.common_grid['lat'].min().values}.")
        
        print('done')

    def make_subset(self,destination_grid):
        subset_lat_bool = (self.common_grid['lat'] > destination_grid.common_grid['lat'].min()) & (self.common_grid['lat'] < destination_grid.common_grid['lat'].max())
        subset_lon_bool = (self.common_grid['lon'] > destination_grid.common_grid['lon'].min()) & (self.common_grid['lon'] < destination_grid.common_grid['lon'].max())
        return subset_lat_bool,subset_lon_bool
        return self.common_grid.isel([subset_lat_bool & subset_lon_bool])

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
            ds_in=self.common_grid,
            ds_out=destination_grid.common_grid,
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
    src_grid = GRID("/projectsa/NEMO/joncon/pyIC_data/src_domain_cfg.nc",ds_lon_name = 'glamt', ds_lat_name = 'gphit')
    dst_grid = GRID("/projectsa/NEMO/joncon/pyIC_data/dst_domain_cfg.nc",ds_lon_name="x",ds_lat_name="y")
    a,b = src_grid.is_superset_of(dst_grid)
    small = src_grid.common_grid.where(a&b)
    #src_grid.regrid(dst_grid)