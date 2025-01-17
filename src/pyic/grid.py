import copy as cp
import warnings

import cf_xarray
import numpy as np
import xarray as xr


class GRID:
    """Class that provides methods to handle and regrid gridded datasets for NEMO."""

    def open_dataset(self, filename, convert_to_z, z_kwargs):
        """Open a dataset from a specified filename using xarray.

        Args:
            filename (str): The path to the dataset file.
            convert_to_z (bool)
            zkawrgs (dict)

        Returns:
            xarray.Dataset: The opened dataset.
        """
        if convert_to_z:
            return self.convert_grid(filename)
        return xr.open_dataset(filename)  # Use xarray to open the dataset file

    def get_dim_varname(self, dimtype):
        """Retrieve the variable name corresponding to a specified dimension type (longitude or latitude).

        Args:
            dimtype (str): The type of dimension ('longitude' or 'latitude').

        Returns:
            str: The variable name for the specified dimension.

        Raises:
            Exception: If the variable name for the specified dimension is not found in the dataset.
        """
        if dimtype == "longitude":
            matches = self.lon_names
        elif dimtype == "latitude":
            matches = self.lat_names

        # Check if any of the potential names exist in the dataset
        for var in matches:
            if var in self.ds:
                return var

        # Raise an exception if no variable name is found
        raise Exception(
            f"Missing variable name for {dimtype} in data set. I tried these {matches} already."
            + f" Specify the variable name for {dimtype} explicitly using the"
            + f"'ds_{dimtype[:3]}_name' argument."
        )

    def extract_lonlat(self, lon_name=None, lat_name=None):
        """Extract longitude and latitude data arrays from the dataset.

        Args:
            lon_name (str, optional): The name of the longitude variable. If None, it will be inferred.
            lat_name (str, optional): The name of the latitude variable. If None, it will be inferred.

        Returns:
            tuple: A tuple containing the longitude DataArray, latitude DataArray, and their respective names.

        Raises:
            Exception: If the specified longitude or latitude variable is not found in the dataset.
        """
        # Extract longitude data
        if lon_name is not None:
            if lon_name in self.ds:
                lon_da = self.ds[lon_name]  # Get the longitude DataArray from the dataset
            else:
                raise Exception(f"{lon_name} not in given data set.")  # Raise an error if not found
        else:
            lon_name = self.get_dim_varname("longitude")  # Infer the longitude variable name
            lon_da = self.ds[lon_name]  # Get the longitude DataArray

        # Extract latitude data
        if lat_name is not None:
            if lat_name in self.ds:
                lat_da = self.ds[lat_name]  # Get the latitude DataArray from the dataset
            else:
                raise Exception(f"{lat_name} not in given data set.")  # Raise an error if not found
        else:
            lat_name = self.get_dim_varname("latitude")  # Infer the latitude variable name
            lat_da = self.ds[lat_name]  # Get the latitude DataArray

        # If longitude or latitude is 1D, create a meshgrid for 2D representation
        if lon_da.ndim == 1 and lat_da.ndim == 1:
            lon_arr, lat_arr = np.meshgrid(lon_da, lat_da)
            lon_da = xr.DataArray(lon_arr, dims=["y", "x"])  # Create a 2D meshgrid for longitude
            lat_da = xr.DataArray(lat_arr, dims=["y", "x"])  # Create a 2D meshgrid for latitude

            self.ds[lat_name] = lat_da
            self.ds[lon_name] = lon_da

        return (
            lon_da,
            lat_da,
            lon_name,
            lat_name,
        )  # Return the longitude and latitude DataArrays and their names

    def make_common_coords(self, lon_name, lat_name, time_counter="time_counter", convert_to_z_grid=False):
        """Align the grid dataset with common coordinate names for regridding.

        Args:
            lon_name (str): The name of the longitude coordinate.
            lat_name (str): The name of the latitude coordinate.
            time_counter (str, optional): The name of the time counter variable. Defaults to "time_counter".

        Returns:
            xarray.Dataset: The dataset with standardized coordinate names and attributes for regridding.
        """
        ds_grid = self.ds.rename({lon_name: "lon", lat_name: "lat"})

        # Assign attributes to the latitude variable for clarity and standardization
        ds_grid["lat"] = ds_grid["lat"].assign_attrs(units="degrees_north", standard_name="latitude")
        # Assign attributes to the longitude variable for clarity and standardization
        ds_grid["lon"] = ds_grid["lon"].assign_attrs(units="degrees_east", standard_name="longitude")

        # Set the latitude and longitude variables as coordinates in the dataset
        ds_grid = ds_grid.set_coords(("lat", "lon"))
        # Add bounds to the latitude and longitude coordinates for better spatial representation
        ds_grid = ds_grid.cf.add_bounds(keys=["lon", "lat"])

        # Check if the time_counter variable exists in the dataset
        if time_counter in ds_grid:
            # If it exists, select the first time step and rename the longitude and latitude variables
            ds_grid_t0 = ds_grid.isel({time_counter: 0})
        else:
            # If it doesn't exist, simply rename the longitude and latitude variables
            ds_grid_t0 = ds_grid

        return ds_grid_t0, ds_grid  # Return the modified dataset with common coordinates

    def convert_grid(filename, z_kwargs):
        """TODO. Vertical regrid of data.

        using xgcm's built in vertical regridder with xCDAT.
        """
        # xcdat documentation here https://xcdat.readthedocs.io/en/main-doc-fix/examples/regridding-vertical.html
        # xgcm documentation here https://xgcm.readthedocs.io/en/latest/transform.html?highlight=vertical

        import xcdat

        ds_grid = xr.open_dataset(filename)

        if "lev" not in z_kwargs:
            raise Exception("Provide z levels to regrid to using z_kwargs = {'lev':[some levels]}.")
        if "var" not in z_kwargs:
            raise Exception("Provide origin vertical grid variable as z_kwargs = {'var':'so'}.")
        if "method" not in z_kwargs:
            method = z_kwargs["method"]
        else:
            method = "linear"

        ds_grid = ds_grid.regridder.vertical(z_kwargs["var"], z_kwargs["lev"], tool="xgcm", method=method)

        return ds_grid

    def __init__(
        self,
        data_filename=None,
        ds_lon_name=None,
        ds_lat_name=None,
        ds_time_counter="time_counter",
        convert_to_z_grid=False,
        z_kwargs={},
    ):
        """Initialize the GRID class with the specified dataset and coordinate names.

        Args:
            data_filename (str, optional): Path to the dataset file on the desired grid.
            ds_lon_name (str, optional): The name of the longitude variable in the dataset.
                                          If None, it will be inferred from common names.
            ds_lat_name (str, optional): The name of the latitude variable in the dataset.
                                          If None, it will be inferred from common names.
            ds_time_counter (str, optional): The name of the time counter variable in the dataset.
                                              If None, it will be inferred from common names.
            convert_to_z_grid (bool, optional): whether to convert from a sigma-level grid to
                                                a z-level grid.
            z_kwargs (dict, optional): additional details required for vertical regridding
        """
        self.data_filename = data_filename  # Store the path to the dataset file
        self.lon_names = ["glamt", "nav_lon"]  # List of potential longitude variable names
        self.lat_names = ["gphit", "nav_lat"]  # List of potential latitude variable names

        # Open the dataset using the provided filename
        self.ds = self.open_dataset(self.data_filename, convert_to_z_grid, z_kwargs)

        # Extract longitude and latitude DataArrays and their names
        self.lon, self.lat, ds_lon_name, ds_lat_name = self.extract_lonlat(ds_lon_name, ds_lat_name)

        # Create a common grid with standardized coordinate names
        self.common_grid, self.common_ds = self.make_common_coords(
            ds_lon_name, ds_lat_name, ds_time_counter, convert_to_z_grid
        )

        # Store the names of the longitude and latitude variables for later use
        self.coords = {"lon_name": ds_lon_name, "lat_name": ds_lat_name}

        # Initialize additional attributes for later processing
        self.inset = None  # Placeholder for inset data
        self.inset_ds = None
        self.lon_bool, self.lat_bool = None, None  # Boolean flags for longitude and latitude checks


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
                np.vstack((arr_in[ind_e], arr_in[ind_w], arr_in[ind_n], arr_in[ind_s])),
                axis=0,
            )

        # Find new indices for next loop
        ind = np.where(np.logical_and(np.isnan(arr_in), np.greater(bathy, 0.0)))
        counter += 1

    return arr_in
