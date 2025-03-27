<!-- markdownlint-disable -->

<a href="../../src/pyic/regrid.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `regrid`

Group of functions to regrid GRIDs to other variations.

---

<a href="../../src/pyic/regrid.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_subset_and_mask`

```python
make_subset_and_mask(source_grid, destination_grid)
```

Create a subset of the source grid that matches the area of the destination grid.

**Args:**

- <b>`source_grid`</b> (GRID): The source grid instance.
- <b>`destination_grid`</b> (GRID): The destination grid instance.

---

<a href="../../src/pyic/regrid.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `subset_mask`

```python
subset_mask(source_grid, destination_grid, return_masks=False)
```

Create a mask for the source grid that covers the same area as the destination grid.

**Args:**

- <b>`source_grid`</b> (GRID): The source grid instance.
- <b>`destination_grid`</b> (GRID): The destination grid instance.
- <b>`return_masks`</b> (bool): If True, return the masks for longitude and latitude.

**Returns:**

- <b>`tuple`</b>: Longitude and latitude masks if return_masks is True.

---

<a href="../../src/pyic/regrid.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_subset`

```python
make_subset(source_grid, subset_lon_bool=None, subset_lat_bool=None)
```

Create a subset of the source grid based on the provided longitude and latitude masks.

**Args:**

- <b>`source_grid`</b> (GRID): The source grid instance.
- <b>`subset_lon_bool`</b> (array-like, optional): Boolean mask for longitude.
- <b>`subset_lat_bool`</b> (array-like, optional): Boolean mask for latitude.

**Returns:**

- <b>`xarray.Dataset`</b>: The subsetted dataset.

---

<a href="../../src/pyic/regrid.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_superset_of`

```python
is_superset_of(source_grid, destination_grid, return_indices=False, tolerance=0)
```

Check if the source grid is a superset of the destination grid.

**Args:**

- <b>`source_grid`</b> (GRID): The source grid instance.
- <b>`destination_grid`</b> (GRID): The destination grid instance.
- <b>`return_indices`</b> (bool): If True, return indices of the source grid for insetting.
- <b>`tolerance`</b> (float): Tolerance for checking bounds.

**Returns:**

- <b>`tuple`</b>: Indices of the source grid if return_indices is True.

**Raises:**

- <b>`Exception`</b>: If the source grid is not a superset of the destination grid.

---

<a href="../../src/pyic/regrid.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_regridder`

```python
make_regridder(
    source_grid,
    destination_grid,
    landsea_mask=None,
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
)
```

Create a regridder to transform the source grid onto the destination grid.

**Args:**

- <b>`source_grid`</b> (GRID): The source grid instance.
- <b>`destination_grid`</b> (GRID): The destination grid instance.
- <b>`landsea_mask`</b> (str): variable name of the landsea mask.
- <b>`regrid_algorithm`</b> (str): The regridding method to use (e.g., "bilinear", "conservative").
- <b>`save_weights`</b> (str, optional): Path to save the regridding weights.
- <b>`reload_weights`</b> (str, optional): Path to load existing regridding weights.
- <b>`periodic`</b> (bool): If True, allows periodic boundaries in the regridding process.
- <b>`ignore_degenerate`</b> (bool): If True, ignores degenerate grid cells during regridding.
- <b>`unmapped_to_nan`</b> (bool): If True, sets unmapped values to NaN in the output.
- <b>`force`</b> (bool): If True, skip superset checks and force regridding (equivalent to setting both check_superset and use_inset to False).
- <b>`check_superset`</b> (bool): If True, check source is a superset of destination.
- <b>`use_inset`</b> (bool): If True, make inset of source. Sometimes results in a type error.

**Returns:**

- <b>`xesmf.Regridder`</b>: The regridder object for transforming data.

---

<a href="../../src/pyic/regrid.py#L227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `vertical_regrid`

```python
vertical_regrid(dataset, vertical_coord, levels, method="linear", kwargs={})
```

Vertically regrid the dataset.

Regrid onto specified levels using preferred method of regridding (wraps xarray.Dataset.interp). https://docs.xarray.dev/en/stable/generated/xarray.Dataset.interp.html

**Args:**

- <b>`dataset`</b> (xarray.Dataset): object to be verticaly regridded
- <b>`vertical_coord`</b> (str): coordinate name of the vertical.
- <b>`levels`</b> (array_like): levels to interpolate Dataset onto.
- <b>`method`</b> (str): interpolation method (see xr documentation for more info).
- <b>`kwargs`</b> (dict): other arguments to pass to xarray.Dataset.interp.

**Returns:**
regridded xarray.Dataset object.

---

<a href="../../src/pyic/regrid.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `infill`

```python
infill(arr_in, n_iter=None, bathy=None)
```

Floodfill missing data.

Returns data with any NaNs replaced by iteratively taking the geometric mean of surrounding points until all NaNs are removed or n_inter-ations have been performed. Input data must be 2D and can include a bathymetry array as to provide land barriers to the infilling.

**Args:**

- <b>`arr_in`</b> (ndarray): data array 2D
- <b>`n_iter`</b> (int): number of smoothing iterations
- <b>`bathy`</b> (ndarray): bathymetry array (land set to zero)

**Returns:**

- <b>`arr_mod`</b> (ndarray): modified data array

---

<a href="../../src/pyic/regrid.py#L316"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_wet_points_populated`

```python
test_wet_points_populated(regridded_ds, dest_mask)
```

Test that wet points have been populated after regridding.

**Args:**

- <b>`regridded_ds`</b> (xarray.Dataset): regridded dataset object
- <b>`dest_mask`</b> (ndarray): landsea mask for destination grid

**Returns:**

- <b>`regridded_ds`</b> (xarray.Dataset): regridded dataset object, infill function used if data are missing.

---

<a href="../../src/pyic/regrid.py#L333"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `regrid_data`

```python
regrid_data(
    source_data,
    dest_grid=None,
    regridder=None,
    regrid_vertically=False,
    vertical_kwargs={},
    dest_grid_mask=None,
)
```

Regrid the source data onto the destination grid using the specified regridder.

One of dest_grid or regridder must be provided. If no regridder provided then one is made using the dest_grid.

**Args:**

- <b>`source_data`</b> (GRID): The source data instance.
- <b>`dest_grid`</b> (GRID, optional): The destination grid instance.
- <b>`regridder`</b> (xesmf.Regridder, optional): The regridder object to use.
- <b>`regrid_vertically`</b> (bool,optional): whether to regrid vertically
- <b>`vertical_kwargs`</b> (dict, optional): dict containing arguments for vertical_regrid function. Must contain "vertical_coord" and "levels" as a minimum.

**Returns:**

- <b>`xarray.Dataset`</b>: The regridded data.

**Raises:**

- <b>`Exception`</b>: If neither dest_grid nor regridder is provided.

---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
