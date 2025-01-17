import numpy as np
import xarray as xr


# Function to generate dummy salinity and temperature data
def generate_dummy_data(num_samples=100, num_lat=5, num_lon=5, num_depth=10):
    # Create a time dimension
    time = np.arange(num_samples)

    # Create depth levels (e.g., from 0 to 1000 meters)
    depth = np.linspace(0, 1000, num_depth)  # Depth from 0 to 1000 meters

    # Generate random salinity values (in practical range, e.g., 0 to 40 ppt)
    # Simulate a gradient: salinity increases with depth
    salinity = np.zeros((num_samples, num_depth, num_lat, num_lon))
    for d in range(num_depth):
        salinity[:, d, :, :] = np.random.uniform(30, 35, (num_samples, num_lat, num_lon)) + (
            d * 0.5
        )  # Gradual increase with depth

    # Generate random temperature values (in practical range, e.g., -2 to 30 degrees Celsius)
    # Simulate a gradient: temperature decreases with depth
    temperature = np.zeros((num_samples, num_depth, num_lat, num_lon))
    for d in range(num_depth):
        temperature[:, d, :, :] = np.random.uniform(20, 25, (num_samples, num_lat, num_lon)) - (
            d * 0.2
        )  # Gradual decrease with depth

    # Create latitude and longitude coordinates
    latitudes = np.linspace(-90, 90, num_lat)  # From -90 to 90 degrees
    longitudes = np.linspace(-179, 179, num_lon)  # From -180 to 180 degrees

    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            "salinity": (("time", "depth", "latitude", "longitude"), salinity),
            "temperature": (("time", "depth", "latitude", "longitude"), temperature),
        },
        coords={"time": time, "depth": depth, "latitude": latitudes, "longitude": longitudes},
    )

    return ds


if __name__ == "__main__":
    # Specify the number of samples, latitudes, longitudes, and depths
    num_samples = 3
    num_lat = 60
    num_lon = 60
    num_depth = 12

    # Generate the data
    dummy_data = generate_dummy_data(num_samples, num_lat, num_lon, num_depth)

    # Optionally, save the dataset to a NetCDF file
    dummy_data.to_netcdf(
        "~/pyIC/examples/synthetic_data/1d/1D_salinity_temperature_data.nc"
    )
