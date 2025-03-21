# regrid.py

This file contains useful functions for regridding.

This is usually a two step process:

1. Read the source and destination grids in using the GRID class.
1. Make a regridder using `make_regridder` to regrid from source GRID to destination GRID.
1. Pass `regrid_data` the regridder and data on the same grid as the source GRID.

You can now save the regridded data as a netcdf, and pass this as an option to `regrid_data`.
