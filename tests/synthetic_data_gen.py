# based off IMMERSE_test_cases.ipynb
# customised version of the file in examples/ used here for testing.

# Import packages
from typing import Union

import numpy as np
import xarray as xr
import xesmf as xe


def rmax(dept):
    rmax_x = np.zeros_like(dept)
    rmax_y = np.zeros_like(dept)

    rmax_x[:, 1:-1] = 0.5 * (
        np.diff(dept[:, :-1], axis=1) / (dept[:, :-2] + dept[:, 1:-1])
        + np.diff(dept[:, 1:], axis=1) / (dept[:, 1:-1] + dept[:, 2:])
    )
    rmax_y[1:-1, :] = 0.5 * (
        np.diff(dept[:-1, :], axis=0) / (dept[:-2, :] + dept[1:-1, :])
        + np.diff(dept[1:, :], axis=0) / (dept[1:-1, :] + dept[2:, :])
    )

    rmax = np.maximum(np.abs(rmax_x), np.abs(rmax_y))

    return rmax


def channel(lx=390.0, ly=294.0, dx=6.0, dy=6.0, stiff=1, H0=4500.0):
    """Create a channel idealised domain.

    This function produces bathymetry of a channel with a Gaussian
    seamount in order to simulate an idealised test case.

    Args:
        lx, ly     (float): dimensions of the domain (km)
        dx, dy     (float): length of grid cells (km)
        stiff      (float): scale factor for steepness of seamount

    Returns:
        zt    (np.ndarray): depth array at t-points (m)
        rm    (np.ndarray): r-max array at t-points (m)
        x, y  (np.ndarray): coordinates (km)
    """
    # Work out the number of grid points in I/J
    nx, ny = np.rint(lx / dx), np.rint(ly / dy)
    nx_int, ny_int = nx.astype(np.int32), ny.astype(np.int32)

    # Redefine cell size to factor into lx and ly
    dx, dy = lx / nx, ly / ny

    # Define T-grid
    x = np.linspace(0.0 + dx / 2.0, lx - dx / 2.0, num=nx_int)
    y = np.linspace(0.0 + dy / 2.0, ly - dy / 2.0, num=ny_int)
    x, y = np.meshgrid(x, y)

    # Define Gaussian bowl and R-max
    zt = np.ma.array(
        H0 * (1.0 - 0.9 * np.exp(-(stiff / 40.0**2 * ((x - lx / 2.0) ** 2 + (y - ly / 2.0) ** 2)))),
        mask=False,
    )
    rm = np.ma.array(rmax(zt), mask=False)

    return x, y, zt, rm


def slope(lx=500.0, ly=500.0, dx=10.0, dy=10.0, lx2=50.0, ly2=100.0, stiff=1):
    """Create an idealised sloped domain.

    This function produces sloped bathymetry in order to
    simulate an idealised overflow test case.

    Args:
        lx, ly     (float): dimensions of the domain (km)
        dx, dy     (float): length of grid cells (km)
        lx2, ly2   (float): dimensions of dense water inlet (km)
        stiff      (float): scale factor for steepness of seamount

    Returns:
        zt    (np.ndarray): depth array at t-points (m)
        rm    (np.ndarray): r-max array at t-points (m)
        x, y  (np.ndarray): coordinates (km)
    """
    # Work out the number of grid points in I/J
    nx, ny = np.rint(lx / dx), np.rint(ly / dy)
    nx_int, ny_int = nx.astype(np.int32), ny.astype(np.int32)

    # Redefine cell size to factor into lx and ly
    dx, dy = lx / nx, ly / ny

    # Add the ledge to nx
    nx2, ny2 = np.floor(lx2 / dx), np.floor(ly2 / dy)
    nx2_int, ny2_int = nx2.astype(np.int32), ny2.astype(np.int32)
    lx2 = nx2 * dx

    # Define T-grid
    x = np.linspace(-lx2 + dx / 2.0, lx - dx / 2.0, num=nx_int + nx2_int)
    y = np.linspace(0.0 + dy / 2.0, ly - dy / 2.0, num=ny_int)
    x, y = np.meshgrid(x, y)

    # Define Gaussian bowl and R-max
    zt = np.ma.array(stiff * 10.0 * x + 600.0, mask=False)
    zt = np.ma.where(zt < 600.0, 600.0, zt)
    zt = np.ma.where(zt > 3600.0, 3600.0, zt)
    rm = np.ma.array(rmax(zt), mask=False)

    # Create source region (i.e. channel where dense water will be released)
    zt.data[:ny_int, :nx2_int] = 0.0
    zt.data[ny_int + ny2_int :, :nx2_int] = 0.0

    return x, y, zt, rm


def half_bowl(lx=500.0, ly=500.0, dx=10.0, dy=10.0, lx2=50.0, ly2=100.0, stiff=1):
    """Create a half bowl idealised domain.

    This function produces a half Gaussian-like bathymetry
    in order to simulate an idealised overflow test case.

    Args:
        lx, ly     (float): dimensions of the domain (km)
        dx, dy     (float): length of grid cells (km)
        lx2, ly2   (float): dimensions of dense water inlet (km)
        stiff      (float): scale factor for steepness of slope

    Returns:
        zt    (np.ndarray): depth array at t-points (m)
        rm    (np.ndarray): r-max array at t-points (m)
        x, y  (np.ndarray): coordinates (km)
    """
    # Work out the number of grid points in I/J
    nx, ny = np.rint(lx / dx), np.rint(ly / dy)
    nx_int, ny_int = nx.astype(np.int32), ny.astype(np.int32)
    mx, my = (nx_int + 1) / 2, (ny_int + 1) / 2

    # Redefine cell size to factor into lx and ly
    dx, dy = lx / nx, ly / ny

    # Add the ledge to nx
    nx2, ny2 = np.floor(lx2 / dx), np.floor(ly2 / dy)
    nx2_int, ny2_int = nx2.astype(np.int32), ny2.astype(np.int32)
    lx2 = nx2 * dx

    # Define T-grid
    x = np.linspace(-lx2 + dx / 2.0, lx - dx / 2.0, num=nx_int + nx2_int)
    y = np.linspace(0.0 + dy / 2.0, ly - dy / 2.0, num=ny_int)
    x, y = np.meshgrid(x, y)

    # Define Gaussian bowl and R-max
    zt = np.ma.array(
        3000.0
        * np.exp(
            -(stiff * 10.0 / (lx / 2.0) ** 2) * (x - lx / 2.0) ** 2
            - (stiff * 10.0 / (ly / 2.0) ** 2) * (y - ly / 2.0) ** 2
        )
        + 600.0,
        mask=False,
    )
    rm = np.ma.array(rmax(zt), mask=False)

    # Open bowl
    mx = int(mx - 0.5)
    zt.data[:, mx + nx2_int :] = zt.data[:, mx + nx2_int - 1 : mx + nx2_int]
    rm.data[:, mx + nx2_int :] = rm.data[:, mx + nx2_int - 1 : mx + nx2_int]

    # Create source region (i.e. channel where dense water will be released)
    my = int(my - 0.5)
    zt.data[:my, :nx2_int] = 0.0
    zt.data[my + ny2_int :, :nx2_int] = 0.0

    return x, y, zt, rm


def temperature_profile(depth: Union[xr.DataArray, float]) -> Union[np.ndarray, float]:
    """Compute the temperature (in °C) at a given depth (in metres).

    Parameters
    ----------
    depth (DataArray or float): Depth in meters (0 to 6000).
    Returns:        numpy array or float: Temperature in °C.
    """
    depth = xr.DataArray(depth) if not isinstance(depth, xr.DataArray) else depth
    # Smooth temperature profile with a thermocline
    surface_temp = 20
    deep_temp = -2
    thermocline_depth = 1000
    # Depth of the thermocline
    scale_depth = 200
    # Sharpness of the thermocline transition
    temperature = deep_temp + (surface_temp - deep_temp) / (
        1 + np.exp((depth - thermocline_depth) / scale_depth)
    )
    return temperature


def salinity_profile(depth: Union[xr.DataArray, float]) -> Union[np.ndarray, float]:
    """Compute the salinity (in PSU) at a given depth (in metres).

    Parameters
    ----------
    depth (DataArray or float): Depth in meters (0 to 6000).

    Returns
    -------
    numpy array or float: Salinity in PSU.
    """
    depth = xr.DataArray(depth) if not isinstance(depth, xr.DataArray) else depth
    # Smooth salinity profile using an exponential model
    surface_salinity = 34.5
    deep_salinity = 32.7
    scale_depth = 1500
    # Characteristic depth for salinity stabilization
    salinity = deep_salinity - (deep_salinity - surface_salinity) * np.exp(-depth / scale_depth)
    return salinity


def make_rand(ldepths, add_rand=True):
    if add_rand:
        return np.random.random(size=(ldepths)) / 4
    return 0


def make_dataset(lx=450.0, ly=500, dx=10.0, dy=10.0, domain="half_bowl", max_depth=3600, z_sep=100, times=2):
    add_rand = False
    domain_types = ["half_bowl", "slope", "channel"]
    if domain not in domain_types:
        raise Exception(f"Domain type '{domain}' not known, only acceptable domains are: {domain_types}.")
    if domain == "half_bowl":
        x, y, zt, rm = half_bowl(lx=lx, ly=ly, dx=dx, dy=dx, stiff=2)
    if domain == "slope":
        x, y, zt, rm = slope(lx=lx, ly=ly, dx=dx, dy=dy)
    if domain == "channel":
        x, y, zt, rm = channel()
    max_depth = max(max_depth, np.max(zt))
    depths = np.arange(0, max_depth, z_sep)
    ds = xr.Dataset(
        coords={"time_counter": np.arange(0, times, 1), "depth": depths, "y": y[:, 0], "x": x[0, :]}
    )

    temps_grid = np.zeros((times, len(depths), y.shape[0], y.shape[1]))
    salinity_grid = np.zeros((times, len(depths), y.shape[0], y.shape[1]))
    for t in range(times):
        temperatures = temperature_profile(depths)
        salinity = salinity_profile(depths)
        for i in range(len(x)):
            for j in range(len(y)):
                temps_grid[t, :, j, i] = np.where(
                    depths < zt[j, i], temperatures + make_rand(len(depths), add_rand=add_rand), 0
                )
        for i in range(len(x)):
            for j in range(len(y)):
                salinity_grid[t, :, j, i] = np.where(
                    depths < zt[j, i], salinity + make_rand(len(depths), add_rand=add_rand), 0
                )
    ds["temperature"] = (("time_counter", "depth", "y", "x"), temps_grid)
    ds["salinity"] = (("time_counter", "depth", "y", "x"), salinity_grid)
    grid = np.meshgrid(x[0] * 0.64 - 140, y[:, 0] * 0.36 - 90)
    ds["lon"] = (["y", "x"], grid[0])
    ds["lat"] = (["y", "x"], grid[1])
    ds = ds.assign_coords({"lon": ds.lon, "lat": ds.lat})
    ds = ds.drop_vars(["x", "y"])
    return ds


def main():
    # Make synthetic data sets using the make_dataset function from synthetic_data_gen
    ds1 = make_dataset(domain="half_bowl")
    ds2 = make_dataset(dx=1.0, dy=1.0, domain="half_bowl")

    # make cropped ds for regridding
    ds1_crop = xr.Dataset()
    for var in ds1:
        ds1_crop[var] = ds1[var][:, :, 5:45, 5:45]

    rgdr = xe.Regridder(ds2, ds1_crop, method="bilinear")
    ds2_rgd = rgdr(ds2)

    # save dses as netCDF
    ds1.to_netcdf("ds1.nc")
    ds1_crop.to_netcdf("ds1_crop.nc")
    ds2.to_netcdf("ds2.nc")
    ds2_rgd.to_netcdf("ds2_rgd.nc")
    ds1_64 = make_dataset(domain="half_bowl", max_depth=4096, z_sep=64)
    ds1_64.to_netcdf("ds1_64.nc")


if __name__ == "__main__":
    main()
