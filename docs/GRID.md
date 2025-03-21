# The GRID class

The GRID class forms the bare bones of pyIC. Gridded data form the means to create the GRID instances.

## Inputs

GRID has the following arguments:

- `data_filename`: a path to a gridded data set readable by `xarray`.
- `dataset`: an `xarray` Dataset.
- `ds_lon_name`: optional str - the variable name for longitude (can be inferred from `self.lon_names`).
- `ds_lat_name`: optional str - the variable name for latitude (can be inferred from `self.lat_names`).
- `ds_z_name`: optional str - the vertical coordinate if it exists.
- `ds_time_counter`: optional str - the variable name for time, inferred as `time_counter` if not provided.
- `convert_to_z_grid`: optional bool - whether to regrid vertically using arguments in `z_kwargs` (uses [xgcm](https://xgcm.readthedocs.io/en/latest/)).
- `z_kwargs`: optional dict - arguments for vertical regridding. Must contain as a minimum `{'variable':'some_variable','target':xr.DataArray/np.array of levels to interpolate to}`.
- `equation_of_state`: optional str - used to check that the equation of state for source and destination data is the same.

You should provide one of `data_filename` or `dataset` for reasons that should be obvious: we need gridded data for the GRID objects.

## Vertical regridding

This is still under construction but uses [xgcm's regridding tool](https://xgcm.readthedocs.io/en/latest/transform.html). You may decide to regrid your data vertically first, and then create a pyIC GRID object, either by saving to a netCDF, or using the generated `xarray` Dataset as an input to the pyIC grid class.
