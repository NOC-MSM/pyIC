"""Group of functions to regrid GRIDs to other variations."""

import warnings

import numpy as np
import xarray as xr
import xesmf as xe


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


def is_superset_of(source_grid, destination_grid, return_indices=False, tolerance=0):
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
        source_grid.common_grid["lat"].max() + tolerance < destination_grid.common_grid["lat"].max()
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
    check_superset=True,
    use_inset=False,
    parallel=False,
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
        force (bool): If True, skip superset checks and force regridding
                      (equivalent to setting both check_superset and use_inset to False).
        check_superset (bool): If True, check source is a superset of destination.
        use_inset (bool): If True, make inset of source. Sometimes results in a type error.

    Returns:
        xesmf.Regridder: The regridder object for transforming data.
    """
    if force or (not (use_inset or check_superset)):
        # If forced, use the entire common grid as the inset
        source_grid.inset = source_grid.common_grid
    else:
        if check_superset:
            # Check if the source grid is a superset of the destination grid
            is_superset_of(source_grid, destination_grid)
        if use_inset:
            # Create masks for the source grid based on the destination grid
            subset_mask(source_grid, destination_grid)
            # Create a subset of the source grid based on the masks
            make_subset(source_grid)
        else:
            source_grid.inset = source_grid.common_grid

    # Create a regridder object using xesmf
    regridder = xe.Regridder(
        ds_in=source_grid.inset,  # Input dataset (subset of the source grid)
        ds_out=destination_grid.common_grid,  # Output dataset (destination grid)
        method=regrid_algorithm,  # Regridding method
        periodic=periodic,  # Allow periodic boundaries
        ignore_degenerate=ignore_degenerate,  # Ignore degenerate grid cells
        unmapped_to_nan=unmapped_to_nan,  # Set unmapped values to NaN
        weights=reload_weights,  # Load weights if specified
        parallel=parallel,  # Whether to create weights in parallel
    )

    # If a path to save weights is provided, save the regridding weights
    if save_weights is not None:
        regridder.to_netcdf(save_weights)

    return regridder  # Return the created regridder


def vertical_regrid(dataset, vertical_coord, levels, method="linear", kwargs={}):
    """Vertically regrid the dataset.

    Regrid onto specified levels using preferred method of regridding (wraps xarray.Dataset.interp).
    https://docs.xarray.dev/en/stable/generated/xarray.Dataset.interp.html

    Args:
        dataset (xr.Dataset): object to be verticaly regridded
        vertical_coord (str): coordinate name of the vertical.
        levels (array_like): levels to interpolate Dataset onto.
        method (str): interpolation method (see xr documentation for more info).
        kwargs (dict): other arguments to pass to xr.Dataset.interp.

    Returns:
        regridded xr.Dataset object.
    """
    if (
        np.min(levels) < dataset[vertical_coord].values.min()
        or np.max(levels) > dataset[vertical_coord].values.max()
    ):
        warnings.warn(
            f"{vertical_coord} levels to interpolate on are outside levels in dataset. Dataset in range "
            f"[{dataset[vertical_coord].values.min()},{dataset[vertical_coord].values.max()}]. "
            f"Provided levels were in the range [{np.min(levels)},{np.max(levels)}]."
            f"\nContinuing anyway, but this may result in peculiar extrapolation."
        )
    regridded = dataset.interp({vertical_coord: levels}, method=method, kwargs=kwargs)
    return regridded


def regrid_data(source_data, dest_grid=None, regridder=None, regrid_vertically=False, vertical_kwargs={}):
    """Regrid the source data onto the destination grid using the specified regridder.

    Args:
        source_data (GRID): The source data instance.
        dest_grid (GRID, optional): The destination grid instance.
        regridder (xesmf.Regridder, optional): The regridder object to use.
        regrid_vertically (bool,optional): whether to regrid vertically
        vertical_kwargs (dict, optional): dict containing arguments for vertical_regrid function.
                                          Must contain "vertical_coord" and "levels" as a minimum.

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
            raise Exception(
                "Provide at least one of dest_grid or regridder."
            )  # Raise an error if neither is provided

    # If the source data's inset is None, use the common grid as the inset
    if source_data.inset_ds is None:
        source_data.inset_ds = source_data.common_grid

    # Use the regridder to transform the inset data to the destination grid
    dest_data = regridder(source_data.inset_ds)

    if regrid_vertically:
        vertical_coord = vertical_kwargs["vertical_coord"]
        levels = vertical_kwargs["levels"]
        if "method" not in vertical_kwargs:
            method = "linear"
        else:
            method = vertical_kwargs["method"]

        extra_kwargs = {}
        for var in vertical_kwargs:
            if var not in ["method", "levels", "vertical_coord"]:
                extra_kwargs[var] = vertical_kwargs[var]
        if vertical_coord in dest_data:
            dest_data = vertical_regrid(
                dest_data, vertical_coord, levels=levels, method=method, kwargs=extra_kwargs
            )
        else:
            raise Exception(f"{vertical_coord} must be in destination.")

    return dest_data  # Return the regridded data
