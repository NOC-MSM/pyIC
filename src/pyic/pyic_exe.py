import argparse

from pyic.grid import GRID
from pyic.regrid import make_regridder, regrid_data


def main():
    parser = argparse.ArgumentParser(
        prog="pyic",
        description="pyIC: Generate initial conditions for regional NEMO configurations out of the box.",
        epilog="Detailed documentation available at https://noc-msm.github.io/pyIC/.",
    )
    parser.add_argument("source", help="path to source grid file")
    parser.add_argument("destination", help="path to destination grid file")
    parser.add_argument("data", help="path to data to be regridded")
    args = parser.parse_args()
    grid1 = GRID(
        data_filename=args.source,
        ds_lon_name="glamt",
        ds_lat_name="gphit",
        ds_time_counter="time",
        ds_z_name="z",
    )
    grid2 = GRID(
        data_filename=args.destination,
        ds_time_counter="t",
        ds_lat_name="gphit",
        ds_lon_name="glamt",
        ds_z_name="z",
    )
    regridder = make_regridder(grid1, grid2, save_weights="weights.nc", landsea_mask="tmask")
    grid1_regrid = regrid_data(args.data, regridder=regridder)
    grid1_regrid.to_netcdf("regridded.nc")


if __name__ == "__main__":
    main()
