"""Group of functions to regrid GRIDs to other variations"""

import xesmf as xe


def regrid(
    source_grid,
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
        ds_in=source_grid.common_grid,
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
