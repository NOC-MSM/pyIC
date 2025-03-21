<!-- markdownlint-disable -->

<a href="../src/pyic/grid.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `grid`






---

<a href="../src/pyic/grid.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GRID`
Class that provides methods to handle and regrid gridded datasets for NEMO. 

<a href="../src/pyic/grid.py#L254"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    data_filename=None,
    dataset=None,
    ds_lon_name=None,
    ds_lat_name=None,
    ds_z_name=None,
    ds_time_counter='time_counter',
    convert_to_z_grid=False,
    z_kwargs={},
    equation_of_state=None
)
```

Initialize the GRID class with the specified dataset and coordinate names. 



**Args:**
 
 - <b>`data_filename`</b> (str, optional):  Path to the dataset file on the desired grid. 
 - <b>`dataset`</b> (xr.Dataset, optional):  xarray Dataset object 
 - <b>`ds_lon_name`</b> (str, optional):  The name of the longitude variable in the dataset.  If None, it will be inferred from common names. 
 - <b>`ds_lat_name`</b> (str, optional):  The name of the latitude variable in the dataset.  If None, it will be inferred from common names. 
 - <b>`ds_z_name`</b> (str, optional):  The name of the depth coordinate, assume z. 
 - <b>`ds_time_counter`</b> (str, optional):  The name of the time counter variable in the dataset,  assume time_counter. 
 - <b>`convert_to_z_grid`</b> (bool, optional):  whether to convert from a sigma-level grid to  a z-level grid. 
 - <b>`z_kwargs`</b> (dict, optional):  additional details required for vertical conversion 
 - <b>`equation_of_state`</b> (str, optional):  the equation of state of the data. 




---

<a href="../src/pyic/grid.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `extract_lonlat`

```python
extract_lonlat(lon_name=None, lat_name=None)
```

Extract longitude and latitude data arrays from the dataset. 



**Args:**
 
 - <b>`lon_name`</b> (str, optional):  The name of the longitude variable. If None, it will be inferred. 
 - <b>`lat_name`</b> (str, optional):  The name of the latitude variable. If None, it will be inferred. 



**Returns:**
 
 - <b>`tuple`</b>:  A tuple containing the longitude DataArray, latitude DataArray, and their respective names. 



**Raises:**
 
 - <b>`Exception`</b>:  If the specified longitude or latitude variable is not found in the dataset. 

---

<a href="../src/pyic/grid.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_dim_varname`

```python
get_dim_varname(dimtype)
```

Retrieve the variable name corresponding to a specified dimension type (longitude or latitude). 



**Args:**
 
 - <b>`dimtype`</b> (str):  The type of dimension ('longitude' or 'latitude'). 



**Returns:**
 
 - <b>`str`</b>:  The variable name for the specified dimension. 



**Raises:**
 
 - <b>`Exception`</b>:  If the variable name for the specified dimension is not found in the dataset. 

---

<a href="../src/pyic/grid.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `make_common_coords`

```python
make_common_coords(z_name, lon_name, lat_name, time_counter='time_counter')
```

Align the grid dataset with common coordinate names for regridding. 



**Args:**
 
 - <b>`z_name`</b> (str):  name of the depth coordinate. 
 - <b>`lon_name`</b> (str):  The name of the longitude coordinate. 
 - <b>`lat_name`</b> (str):  The name of the latitude coordinate. 
 - <b>`time_counter`</b> (str, optional):  The name of the time counter variable. Defaults to "time_counter". 



**Returns:**
 
 - <b>`xarray.Dataset`</b>:  The dataset with standardized coordinate names and attributes for regridding. 

---

<a href="../src/pyic/grid.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `open_dataset`

```python
open_dataset(filename, convert_to_z, z_kwargs)
```

Open a dataset from a specified filename using xarray. 



**Args:**
 
 - <b>`filename`</b> (str):  The path to the dataset file. convert_to_z (bool) zkawrgs (dict) 



**Returns:**
 
 - <b>`xarray.Dataset`</b>:  The opened dataset. 

---

<a href="../src/pyic/grid.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `vertical_convert`

```python
vertical_convert(ds_grid, z_kwargs, periodic=False)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
