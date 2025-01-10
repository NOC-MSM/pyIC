import copy as cp  # Import the copy module to create deep copies of objects
import warnings  # Import the warnings module to manage warning messages

import numpy as np  # Import NumPy for numerical operations and array handling
import xarray as xr  # Import xarray for working with labeled multi-dimensional arrays


class GRID:
    """Class that provides methods to handle and regrid gridded datasets for NEMO."""

    def open_dataset(self, filename):
        """Open a dataset from a specified filename using xarray.

        Args:
            filename (str): The path to the dataset file.

        Returns:
            xarray.Dataset: The opened dataset.
        """
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
            matches = self.lon_names  # Get potential names for longitude variables
        elif dimtype == "latitude":
            matches = self.lat_names  # Get potential names for latitude variables

        # Check if any of the potential names exist in the dataset
        for var in matches:
            if var in self.ds:  # If a match is found
                return var  # Return the found variable name

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
        if len(lon_da.shape) == 1:
            lon_da = xr.DataArray(
                np.meshgrid(lon_da, lon_da), dims=["y", "x"]
            )  # Create a 2D meshgrid for longitude
        if len(lat_da.shape) == 1:
            lat_da = xr.DataArray(
                np.meshgrid(lat_da, lat_da), dims=["y", "x"]
            )  # Create a 2D meshgrid for latitude

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
        # Check if the time_counter variable exists in the dataset
        if time_counter in self.ds:
            # If it exists, select the first time step and rename the longitude and latitude variables
            ds_grid = self.ds.isel({time_counter: 0}).rename({lon_name: "lon", lat_name: "lat"})
        else:
            # If it doesn't exist, simply rename the longitude and latitude variables
            ds_grid = self.ds.rename({lon_name: "lon", lat_name: "lat"})

        # Assign attributes to the latitude variable for clarity and standardization
        ds_grid["lat"] = ds_grid["lat"].assign_attrs(units="degrees_north", standard_name="latitude")
        # Assign attributes to the longitude variable for clarity and standardization
        ds_grid["lon"] = ds_grid["lon"].assign_attrs(units="degrees_east", standard_name="longitude")

        # Set the latitude and longitude variables as coordinates in the dataset
        ds_grid = ds_grid.set_coords(("lat", "lon"))
        # Add bounds to the latitude and longitude coordinates for better spatial representation
        ds_grid = ds_grid.cf.add_bounds(keys=["lon", "lat"])

        return ds_grid  # Return the modified dataset with common coordinates

    def __init__(
        self,
        data_filename=None,
        ds_lon_name=None,
        ds_lat_name=None,
        ds_time_counter="time_counter",
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
        """
        self.data_filename = data_filename  # Store the path to the dataset file
        self.lon_names = ["glamt", "nav_lon"]  # List of potential longitude variable names
        self.lat_names = ["gphit", "nav_lat"]  # List of potential latitude variable names

        # Open the dataset using the provided filename
        self.ds = self.open_dataset(self.data_filename)

        # Extract longitude and latitude DataArrays and their names
        self.lon, self.lat, ds_lon_name, ds_lat_name = self.extract_lonlat(ds_lon_name, ds_lat_name)

        # Create a common grid with standardized coordinate names
        self.common_grid = self.make_common_coords(ds_lon_name, ds_lat_name, ds_time_counter)

        # Store the names of the longitude and latitude variables for later use
        self.coords = {"lon_name": ds_lon_name, "lat_name": ds_lat_name}

        # Initialize additional attributes for later processing
        self.inset = None  # Placeholder for inset data
        self.lon_bool, self.lat_bool = None, None  # Boolean flags for longitude and latitude checks

    def make_inset(self, inset_mask):
        """Create an inset dataset based on a provided mask.

        Args:
            inset_mask (xarray.Dataset): A mask dataset that defines the area to be included in the inset.
        """
        in1 = xr.Dataset()  # Create an empty xarray Dataset for the inset

        # Loop through each variable in the common grid
        for var in self.common_grid:
            # Create a new variable in the inset dataset, applying the mask to drop NaN values
            in1[var] = self.common_grid[var].where(inset_mask[var].notnull(), drop=True)

        # Add bounds to the inset dataset for better spatial representation
        in1 = in1.cf.add_bounds(keys=["lon", "lat"])
        self.inset = in1  # Store the created inset dataset

    def infill(arr_in, n_iter=None, bathy=None):
        """TODO Replace NaN values in 2D array by iteratively taking the geometric mean of surrounding points.

        This method continues until all NaNs are filled or a specified number of iterations is reached.

        Args:
            arr_in (ndarray): Input 2D data array that may contain NaN values.
            n_iter (int, optional): Maximum number of iterations to perform. If None, will run indefinitely
                                    until all NaNs are filled.

            bathy (ndarray, optional): A 2D bathymetry array that indicates land barriers
                                       (land areas should be set to zero).

        Returns:
            ndarray: The modified data array with NaN values replaced.

        Raises:
            ValueError: If the input array does not have two dimensions.
        """
        # Check if the input array has two dimensions
        if arr_in.ndim != 2:
            raise ValueError("Array must have two dimensions")  # Raise an error if not

        # Initialize the bathymetry array if not provided
        if bathy is None:
            bathy = np.ones_like(arr_in, dtype=float)  # Create a default bathymetry array of ones
        if n_iter is None:
            n_iter = np.inf  # Set to infinite iterations if not specified

        # Find indices of NaN values in the input array where bathymetry is greater than zero
        ind = np.where(np.logical_and(np.isnan(arr_in), np.greater(bathy, 0.0)))
        counter = 0  # Initialize a counter for the number of iterations
        jpj, jpi = arr_in.shape  # Get the shape of the input array (number of rows and columns)

        # Infill until all NaNs are removed or the maximum number of iterations is reached
        while np.sum(ind) > 0 and counter < n_iter:
            # Create deep copies of the indices of NaN values for neighboring points
            ind_e = cp.deepcopy(ind)  # Indices for the eastern neighbor
            ind_w = cp.deepcopy(ind)  # Indices for the western neighbor
            ind_n = cp.deepcopy(ind)  # Indices for the northern neighbor
            ind_s = cp.deepcopy(ind)  # Indices for the southern neighbor

            # Update indices to point to neighboring cells, ensuring they stay within bounds
            ind_e[1][:] = np.minimum(ind_e[1][:] + 1, jpi - 1)  # Move east
            ind_w[1][:] = np.maximum(ind_w[1][:] - 1, 0)  # Move west
            ind_n[0][:] = np.minimum(ind_n[0][:] + 1, jpj - 1)  # Move north
            ind_s[0][:] = np.maximum(ind_s[0][:] - 1, 0)  # Move south

            # Replace NaN values in the input array with the mean of neighboring values
            with warnings.catch_warnings():  # Suppress warnings during the operation
                warnings.simplefilter("ignore", category=RuntimeWarning)  # Ignore runtime warnings
                arr_in[ind] = np.nanmean(
                    np.vstack(
                        (arr_in[ind_e], arr_in[ind_w], arr_in[ind_n], arr_in[ind_s])
                    ),  # Stack neighboring values
                    axis=0,  # Calculate the mean along the vertical axis
                )

            # Find new indices of NaN values for the next iteration
            ind = np.where(np.logical_and(np.isnan(arr_in), np.greater(bathy, 0.0)))
            counter += 1  # Increment the iteration counter

        return arr_in  # Return the modified array with NaN values filled
