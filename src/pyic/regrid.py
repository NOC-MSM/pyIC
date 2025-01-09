"""Group of functions to regrid GRIDs to other variations."""

import xarray as xr  # Import xarray for handling multi-dimensional arrays
import xesmf as xe  # Import xesmf for regridding operations


def make_subset_and_mask(source_grid, destination_grid):
    """Create a subset of the source grid that matches the area of the destination grid.

    Args:
        source_grid (GRID): The source grid instance.
        destination_grid (GRID): The destination grid instance.
    """
    # Generate masks for longitude and latitude that define the subset area
    lonm, latm = subset_mask(source_grid, destination_grid, return_masks=True)
    # Create the subset of the source grid based on the generated masks
    make_subset(source_grid, lonm, latm)


def subset_mask(source_grid, destination_grid, return_masks=False):
    """Create a mask for the source grid that covers the same area as the destination grid.

    Args:
        source_grid (GRID): The source grid instance.
        destination_grid (GRID): The destination grid instance.
        return_masks (bool): If True, return the masks for longitude and latitude.

    Returns:
        tuple: Longitude and latitude masks if return_masks is True.
    """
    # Create a boolean mask for latitude values within the bounds of the destination grid
    subset_lat_bool = (source_grid.common_grid["lat"] > destination_grid.common_grid["lat"].min()) & (
        source_grid.common_grid["lat"] < destination_grid.common_grid["lat"].max()
    )
    # Create a boolean mask for longitude values within the bounds of the destination grid
    subset_lon_bool = (source_grid.common_grid["lon"] > destination_grid.common_grid["lon"].min()) & (
        source_grid.common_grid["lon"] < destination_grid.common_grid["lon"].max()
    )
    
    # Store the masks in the source grid for later use
    source_grid.lat_bool = subset_lat_bool
    source_grid.lon_bool = subset_lon_bool
    
    # Return the masks if requested
    if return_masks:
        return subset_lon_bool, subset_lat_bool


def make_subset(source_grid, subset_lon_bool=None, subset_lat_bool=None):
    """Create a subset of the source grid based on the provided longitude and latitude masks.

    Args:
        source_grid (GRID): The source grid instance.
        subset_lon_bool (array-like, optional): Boolean mask for longitude.
        subset_lat_bool (array-like, optional): Boolean mask for latitude.

    Returns:
        xarray.Dataset: The subsetted dataset.
    """
    # Use the stored masks if none are provided
    if subset_lat_bool is None:
        subset_lat_bool = source_grid.lat_bool
    if subset_lon_bool is None:
        subset_lon_bool = source_grid.lon_bool

    # Create an inset dataset by applying the masks to the common grid
    inset = source_grid.common_grid.where(subset_lat_bool & subset_lon_bool, drop=True)

    # Create an empty xarray Dataset for the subset
    in1 = xr.Dataset()
    # Loop through each variable in the inset dataset
    for var in inset:
        # Add the variable to the new dataset, dropping NaN values
        in1[var] = inset[var].where(inset[var].notnull(), drop=True)
    
    # Add bounds to the new dataset for better spatial representation
    in1 = in1.cf.add_bounds(keys=["lon", "lat"])
    # Store the created inset dataset in the source grid
    source_grid.inset = in1
    return source_grid.inset  # Return the inset dataset


def is_superset_of(source_grid, destination_grid, return_indices=True, tolerance=0):
    """Check if the source grid is a superset of the destination grid.

    Args:
        source_grid (GRID): The source grid instance.
        destination_grid (GRID): The destination grid instance.
        return_indices (bool): If True, return indices of the source grid for insetting.
        tolerance (float): Tolerance for checking bounds.

    Returns:
        tuple: Indices of the source grid if return_indices is True.
    
    Raises:
        Exception: If the source grid is not a superset of the destination grid.
    """
    # Check minimum latitude bounds
    if (
        source_grid.common_grid["lat"].min() - tolerance > destination_grid.common_grid["lat"].min()
        and destination_grid.common_grid["lat"].min() > -89
    ):
        raise Exception(
            "Source not superset of destination: min latitude. "
            + f"{source_grid.common_grid['lat'].min().values} > "
            + f"{destination_grid.common_grid['lat'].min().values}."
        )
    # Check maximum latitude bounds
    elif (
        source_grid.common_grid["lat"].max() + tolerance <
        destination_grid.common_grid["lat"].max()
        and destination_grid.common_grid["lat"].max() < 89
    ):
        raise Exception(
            "Source not superset of destination: max latitude. "
            + f"{source_grid.common_grid['lat'].max().values} < "
            + f"{destination_grid.common_grid['lat'].max().values}."
        )
    # Check minimum longitude bounds
    elif (
        source_grid.common_grid["lon"].min() - tolerance > destination_grid.common_grid["lon"].min()
        and destination_grid.common_grid["lon"].min() > -179
    ):
        raise Exception(
            "Source not superset of destination: min longitude. "
            + f"{source_grid.common_grid['lon'].min().values} > "
            + f"{destination_grid.common_grid['lon'].min().values}."
        )
    # Check maximum longitude bounds
    elif (
        source_grid.common_grid["lon"].max() + tolerance < destination_grid.common_grid["lon"].max()
        and destination_grid.common_grid["lon"].max() < 179
    ):
        raise Exception(
            "Source not superset of destination: max longitude. "
            + f"{source_grid.common_grid['lon'].max().values} < "
            + f"{destination_grid.common_grid['lon'].max().values}."
        )
    else:
        print("Source grid is a superset of destination grid.")  # Print confirmation if checks pass
        if return_indices:
            # If requested, create and return the subset and mask for the source grid
            return make_subset_and_mask(source_grid, destination_grid)


def make_regridder(
    source_grid,
    destination_grid,
    regrid_algorithm="bilinear",
    save_weights=None,
    reload_weights=None,
    periodic=True,  
    ignore_degenerate=True,  
    unmapped_to_nan=True,  
    force=False,
):
    """Create a regridder to transform the source grid onto the destination grid.

    Args:
        source_grid (GRID): The source grid instance.
        destination_grid (GRID): The destination grid instance.
        regrid_algorithm (str): The regridding method to use (e.g., "bilinear", "conservative").
        save_weights (str, optional): Path to save the regridding weights.
        reload_weights (str, optional): Path to load existing regridding weights.
        periodic (bool): If True, allows periodic boundaries in the regridding process.
        ignore_degenerate (bool): If True, ignores degenerate grid cells during regridding.
        unmapped_to_nan (bool): If True, sets unmapped values to NaN in the output.
        force (bool): If True, skip superset checks and force regridding.

    Returns:
        xesmf.Regridder: The regridder object for transforming data.
    """
    if force:
        # If forced, use the entire common grid as the inset
        source_grid.inset = source_grid.common_grid
    else:
        # Check if the source grid is a superset of the destination grid
        is_superset_of(source_grid, destination_grid)
        # Create masks for the source grid based on the destination grid
        subset_mask(source_grid, destination_grid)
        # Create a subset of the source grid based on the masks
        make_subset(source_grid)

    # Create a regridder object using xesmf
    regridder = xe.Regridder(
        ds_in=source_grid.inset,  # Input dataset (subset of the source grid)
        ds_out=destination_grid.common_grid,  # Output dataset (destination grid)
        method=regrid_algorithm,  # Regridding method
        periodic=periodic,  # Allow periodic boundaries
        ignore_degenerate=ignore_degenerate,  # Ignore degenerate grid cells
        unmapped_to_nan=unmapped_to_nan,  # Set unmapped values to NaN
        weights=reload_weights,  # Load weights if specified
    )
    
    # If a path to save weights is provided, save the regridding weights
    if save_weights is not None:
        regridder.to_netcdf(save_weights)
    
    return regridder  # Return the created regridder


def regrid_data(source_data, dest_grid=None, regridder=None):
    """Regrid the source data onto the destination grid using the specified regridder.

    Args:
        source_data (GRID): The source data instance.
        dest_grid (GRID, optional): The destination grid instance.
        regridder (xesmf.Regridder, optional): The regridder object to use.

    Returns:
        xarray.DataArray: The regridded data.

    Raises:
        Exception: If neither dest_grid nor regridder is provided.
    """
    # If no regridder is provided, create one using the destination grid if available
    if regridder is None:
        if dest_grid is not None:
            regridder = make_regridder(source_data, destination_grid=dest_grid)
        else:
            raise Exception("Provide at least one of dest_grid or regridder.")  # Raise an error if neither is provided

    # If the source data's inset is None, use the common grid as the inset
    if source_data.inset is None:
        source_data.inset = source_data.common_grid
    
    # Use the regridder to transform the inset data to the destination grid
    dest_data = regridder(source_data.inset)
    
    return dest_data  # Return the regridded data
