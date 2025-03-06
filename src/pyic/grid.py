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
        ds = xr.open_dataset(filename)
        if convert_to_z:
            return self.vertical_regrid(ds, z_kwargs=z_kwargs)
        return ds

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

    def make_common_coords(self, lon_name, lat_name, time_counter="time_counter"):
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
        # Check if the time_counter variable exists in the dataset
        if time_counter in ds_grid:
            # If it exists, select the first time step
            ds_grid = ds_grid.isel({time_counter: 0}).set_coords(("lat", "lon"))
        else:
            # If it doesn't exist, simply rename the longitude and latitude variables
            ds_grid = ds_grid.set_coords(("lat", "lon"))

        # Add bounds to the latitude and longitude coordinates for better spatial representation
        try:
            ds_grid = ds_grid.cf.add_bounds(keys=["lon", "lat"])
        except Exception as e:
            print("Couldn't add bounds.")
            print(e)
            print("Continuing anyway.")

        return ds_grid  # Return the modified dataset with common coordinates

    def vertical_convert(self, ds_grid, z_kwargs, periodic=False):
        print("Vertical conversion is still under construction. Use at your own risk.")
        """Vertical conversion of data using xgcm's built in vertical conversion with their `Grid` class.

        For this to work you will need:
        ds_grid: data set on some metric of depth (let's say salinity)
        z_kwargs: dict containing at least {'variable':str/list of strs,'target':array_like}.
                Other arguments are as in the xgcm documentation:
                https://xgcm.readthedocs.io/en/latest/transform.html?highlight=vertical
        periodic: bool, passed to xgcm.Grid.

        returns vertically regridded data set.
        """

        from xgcm import Grid as xgcm_grid

        # save existing grid if required after regridding
        self.raw_ds = ds_grid

        # check z_kwargs are populated and infer if not
        if "variable" not in z_kwargs:
            raise Exception("Provide origin vertical grid variable as z_kwargs = {...,'variable':'so',...}.")
        if "coord" not in z_kwargs:
            default_coord = "lev"
            print(f"source coord not given, using default, {default_coord}.")
            z_kwargs["coord"] = default_coord
        if "target" not in z_kwargs:
            raise Exception(
                "Provide target levels using z_kwargs = {...,'target':np.array/xr.DataArray,...}."
            )
        if "target_variable" in z_kwargs:
            target_data = ds_grid[z_kwargs["target_variable"]]
        else:
            target_data = None
        if "method" in z_kwargs:
            method = z_kwargs["method"]
        else:
            method = "linear"
            print(f"'method' not specified in z_kwargs: using default, {method}.")
        available_methods = ["linear", "log", "conservative"]
        if method not in available_methods:
            raise Exception(f"Cannot use regridding method {method}. Choose one of {available_methods}.")
        # optional argument handling
        for optional_arg in ["mask_edges", "bypass_checks", "suffix"]:
            if optional_arg not in z_kwargs:
                z_kwargs[optional_arg] = None

        # setup xgcm Grid
        xgrid = xgcm_grid(
            ds_grid,
            coords={
                "Z": {"center": z_kwargs["coord"]},
            },
            periodic=periodic,
        )
        # vertical regrid, either for each variable or individual depending on argument.
        ds_out = xr.Dataset()
        if type(z_kwargs["variable"]) is list:
            for var in z_kwargs["variable"]:
                da_grid = xgrid.transform(
                    da=ds_grid[var],
                    axis="Z",
                    target=z_kwargs["target"],
                    target_data=target_data,
                    method=method,
                    mask_edges=z_kwargs["mask_edges"],
                    bypass_checks=z_kwargs["bypass_checks"],
                    suffix=z_kwargs["suffix"],
                )
                ds_out[var] = da_grid
        elif z_kwargs["variable"] == "all":
            for var in ds_grid:
                try:
                    da_grid = xgrid.transform(
                        da=ds_grid[var],
                        axis="Z",
                        target=z_kwargs["target"],
                        target_data=target_data,
                        method=method,
                        mask_edges=z_kwargs["mask_edges"],
                        bypass_checks=z_kwargs["bypass_checks"],
                        suffix=z_kwargs["suffix"],
                    )
                    ds_out[var] = da_grid
                except Exception as e:
                    print(f"Skipping '{var}' because:")
                    print(e)
        else:
            da_grid = xgrid.transform(
                da=ds_grid[z_kwargs["variable"]],
                axis="Z",
                target=z_kwargs["target"],
                target_data=target_data,
                method=method,
                mask_edges=z_kwargs["mask_edges"],
                bypass_checks=z_kwargs["bypass_checks"],
                suffix=z_kwargs["suffix"],
            )
            ds_out[z_kwargs["variable"]] = da_grid
        return ds_out

    def __init__(
        self,
        data_filename=None,
        dataset=None,
        ds_lon_name=None,
        ds_lat_name=None,
        ds_time_counter="time_counter",
        convert_to_z_grid=False,
        z_kwargs={},
    ):
        """Initialize the GRID class with the specified dataset and coordinate names.

        Args:
            data_filename (str, optional): Path to the dataset file on the desired grid.
            dataset (xr.Dataset, optional): xarray Dataset object
            ds_lon_name (str, optional): The name of the longitude variable in the dataset.
                                          If None, it will be inferred from common names.
            ds_lat_name (str, optional): The name of the latitude variable in the dataset.
                                          If None, it will be inferred from common names.
            ds_time_counter (str, optional): The name of the time counter variable in the dataset.
                                              If None, it will be inferred from common names.
            convert_to_z_grid (bool, optional): whether to convert from a sigma-level grid to
                                                a z-level grid.
            z_kwargs (dict, optional): additional details required for vertical conversion
        """
        self.data_filename = data_filename  # Store the path to the dataset file
        self.lon_names = [
            "glamt",
            "nav_lon",
            "lon",
            "longitude",
        ]
        # List of potential longitude variable names
        self.lat_names = ["gphit", "nav_lat", "lat", "latitude"]  # List of potential latitude variable names

        # Open the dataset
        if self.data_filename is not None:
            self.ds = self.open_dataset(self.data_filename, convert_to_z_grid, z_kwargs)
        elif dataset is not None:
            if convert_to_z_grid:
                self.ds = self.vertical_regrid(dataset, z_kwargs)
            else:
                self.ds = dataset
        else:
            raise Exception("Specify one of 'data_filename' or 'dataset'.")

        # Extract longitude and latitude DataArrays and their names
        self.lon, self.lat, ds_lon_name, ds_lat_name = self.extract_lonlat(ds_lon_name, ds_lat_name)

        # Create a common grid with standardized coordinate names
        self.common_grid = self.make_common_coords(
            ds_lon_name,
            ds_lat_name,
            ds_time_counter,
        )

        # Store the names of the longitude and latitude variables for later use
        self.coords = {"lon_name": ds_lon_name, "lat_name": ds_lat_name}

        # Initialize additional attributes for later processing
        self.inset = None  # Placeholder for inset data
        self.inset_ds = None
        self.lon_bool, self.lat_bool = None, None  # Boolean flags for longitude and latitude checks


def infill(arr_in, n_iter=None, bathy=None):
    # taken from https://github.com/NOC-MSM/ORCHESTRA/blob/master/SCRIPTS/under_ice.py"
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
