import copy as cp
import warnings

import numpy as np
import xarray as xr


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
            + f" Specify the variable name for {dimtype} explicitly using the"
            + f"'ds_{dimtype[:3]}_name' argument."
        )

    def extract_lonlat(self, lon_name=None, lat_name=None):
        if lon_name is not None:
            if lon_name in self.ds:
                lon_da = self.ds[lon_name]
            else:
                raise Exception(f"{lon_name} not in given data set.")
        else:
            lon_name = self.get_dim_varname("longitude")
            lon_da = self.ds[lon_name]
        if lat_name is not None:
            if lat_name in self.ds:
                lat_da = self.ds[lat_name]
            else:
                raise Exception(f"{lat_name} not in given data set.")
        else:
            lat_name = self.get_dim_varname("latitude")
            lat_da = self.ds[lat_name]

        if len(lon_da.shape) == 1:
            lon_da = xr.DataArray(np.meshgrid(lon_da, lon_da), dims=["y", "x"])
        if len(lat_da.shape) == 1:
            lat_da = xr.DataArray(np.meshgrid(lat_da, lat_da), dims=["y", "x"])
        return lon_da, lat_da, lon_name, lat_name

    def make_common_coords(self, lon_name, lat_name, time_counter='time_counter'):
        """Put grid onto grid with lon and lat for coordinate names ready for regridding.

        lon_name: given lon coordinate name
        lat_name: given lat coordinate name
        """
        if time_counter in self.ds:
            print('hello')
            ds_grid = self.ds.isel({time_counter:0}).rename(
            {lon_name: "lon", lat_name: "lat"}
            )
        else:
            ds_grid = self.ds.rename({lon_name: "lon", lat_name: "lat"})
        ds_grid['lat'] = ds_grid['lat'].assign_attrs(units='degrees_north',standard_name='latitude')
        ds_grid['lon'] = ds_grid['lon'].assign_attrs(units='degrees_east',standard_name='longitude')
        ds_grid = ds_grid.set_coords(("lat", "lon"))
        ds_grid = ds_grid.cf.add_bounds(keys=['lon','lat'])
        return ds_grid

    def __init__(self, data_filename=None, 
                 ds_lon_name=None, ds_lat_name=None, ds_time_counter="time_counter"):
        """Initialise the class.

        data_filename: path to a data set on the desired grid.
        ds_lon_name: optional, the name in the data set of the longitude variable.
                     If none, will be inferred from common ones.
        ds_lat_name: optional, the name in the data set of the latitude variable.
                     If none, will be inferred from common ones.
        ds_time_counter: optional, the name in the data set of the time_counter variable.
                     If none, will be inferred from common ones.
        """
        self.data_filename = data_filename
        self.lon_names = ["glamt,", "x", "nav_lon"]
        self.lat_names = ["gphit", "y", "nav_lat"]
        self.ds = self.open_dataset(self.data_filename)
        self.lon, self.lat, ds_lon_name, ds_lat_name = self.extract_lonlat(ds_lon_name, ds_lat_name)
        self.common_grid = self.make_common_coords(ds_lon_name, ds_lat_name,ds_time_counter)
        self.coords = {"lon_name": ds_lon_name, "lat_name": ds_lat_name}
        self.inset = None
        # return self.ds,self.lat,self.lon
    
    def make_inset(self,inset_mask):
        in1 = xr.Dataset()
        for var in self.common_grid:
             print(self.common_grid.var.shape,inset_mask[var].shape)
             in1[var] = self.common_grid[var].where(inset_mask[var].notnull(), drop=True)
        in1 = in1.cf.add_bounds(keys=['lon','lat'])
        self.inset = in1

    def infill(arr_in, n_iter=None, bathy=None):
        """TODO: INTEGRATE WITH CLASS PROPERLY.

        Returns data with any NaNs replaced by iteratively taking the geometric
        mean of surrounding points until all NaNs are removed or n_inter-ations
        have been performed. Input data must be 2D and can include a
        bathymetry array as to provide land barriers to the infilling.

        Args:
            arr_in          (ndarray): data array 2D
            n_iter              (int): number of smoothing iterations
            bathy           (ndarray): bathymetry array (land set to zero)

        Returns
        -------
            arr_mod         (ndarray): modified data array
        """
        # Check number of dims
        if arr_in.ndim != 2:
            raise ValueError("Array must have two dimensions")

        # Intial setup to prime things for the averaging loop
        if bathy is None:
            bathy = np.ones_like(arr_in, dtype=float)
        if n_iter is None:
            n_iter = np.inf
        ind = np.where(np.logical_and(np.isnan(arr_in), np.greater(bathy, 0.0)))
        counter = 0
        jpj, jpi = arr_in.shape
        # Infill until all NaNs are removed or N interations completed
        while np.sum(ind) > 0 and counter < n_iter:
            # TODO: include a check to see if number of NaNs is decreasing

            # Create indices of neighbouring points
            ind_e = cp.deepcopy(ind)
            ind_w = cp.deepcopy(ind)
            ind_n = cp.deepcopy(ind)
            ind_s = cp.deepcopy(ind)

            ind_e[1][:] = np.minimum(ind_e[1][:] + 1, jpi - 1)
            ind_w[1][:] = np.maximum(ind_w[1][:] - 1, 0)
            ind_n[0][:] = np.minimum(ind_n[0][:] + 1, jpj - 1)
            ind_s[0][:] = np.maximum(ind_s[0][:] - 1, 0)

            # Replace NaNs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                arr_in[ind] = np.nanmean(
                    np.vstack(
                        (arr_in[ind_e], arr_in[ind_w], arr_in[ind_n], arr_in[ind_s])
                    ),
                    axis=0,
                )

            # Find new indices for next loop
            ind = np.where(np.logical_and(np.isnan(arr_in), np.greater(bathy, 0.0)))
            counter += 1

        return arr_in
