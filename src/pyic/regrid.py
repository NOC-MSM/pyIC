"""Group of functions to regrid GRIDs to other variations"""

import xarray as xr
import xesmf as xe

def make_subset(source_grid, destination_grid):
        subset_lat_bool = (
            source_grid.common_grid["lat"] > destination_grid.common_grid["lat"].min()
        ) & (source_grid.common_grid["lat"] < destination_grid.common_grid["lat"].max())
        subset_lon_bool = (
            source_grid.common_grid["lon"] > destination_grid.common_grid["lon"].min()
        ) & (source_grid.common_grid["lon"] < destination_grid.common_grid["lon"].max())
        inset = source_grid.common_grid.where(subset_lat_bool & subset_lon_bool)
        in1 = xr.Dataset()
        for var in inset:
             in1[var] = inset[var].where(inset[var].notnull(), drop=True)
        in1 = in1.cf.add_bounds(keys=['lon','lat'])
        source_grid.inset = in1
        return source_grid.inset

def is_superset_of(source_grid, destination_grid, return_indices=True, tolerance=0):
        """Check if source grid is a superset of the destination grid"""
        if (
            source_grid.common_grid["lat"].min() - tolerance > destination_grid.common_grid["lat"].min()
            and destination_grid.common_grid["lat"].min() > -89
        ):
            raise Exception(
                f"Source not superset of destination: min latitude. {source_grid.common_grid['lat'].min().values} > {destination_grid.common_grid['lat'].min().values}."
            )
        elif (
            source_grid.common_grid["lat"].max() + tolerance < destination_grid.common_grid["lat"].max()
            and destination_grid.common_grid["lat"].max() < 89
        ):
            raise Exception(
                f"Source not superset of destination: max latitude. {source_grid.common_grid['lat'].max().values} < {destination_grid.common_grid['lat'].max().values}."
            )
        elif (
            source_grid.common_grid["lon"].min() - tolerance > destination_grid.common_grid["lon"].min()
            and destination_grid.common_grid["lon"].min() > -179
        ):
            raise Exception(
                f"Source not superset of destination: min longitude. {source_grid.common_grid['lon'].min().values} > {destination_grid.common_grid['lon'].min().values}."
            )
        elif (
            source_grid.common_grid["lon"].max() + tolerance < destination_grid.common_grid["lon"].max()
            and destination_grid.common_grid["lon"].max() < 179
        ):
            raise Exception(
                f"Source not superset of destination: max longitude. {source_grid.common_grid['lon'].max().values} < {destination_grid.common_grid['lon'].max().values}."
            )
        else:
            print("Source grid is a superset of destination grid.")
            if return_indices:
                return make_subset(source_grid,destination_grid)

def regrid(
    source_grid,
    destination_grid,
    regrid_algorithm="bilinear",
    save_weights=None,
    reload_weights=None,
    force=False
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
    if force:
        source_grid.inset = source_grid.common_grid
    else:
        is_superset_of(source_grid,destination_grid)
        make_subset(source_grid,destination_grid)
    regridder = xe.Regridder(
        ds_in=source_grid.inset,
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
