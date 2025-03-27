import argparse

from pyic.grid import GRID
from pyic.regrid import make_regridder, regrid_data


def main():
    """Call pyic from the command line for simple regridding tasks.

    Requires arguments for the source and destination grids,
    and the data to be regridded from the source to the destination grid.

    """
    parser = argparse.ArgumentParser(
        prog="pyic",
        description="pyIC: Generate initial conditions for regional NEMO configurations out of the box.",
        epilog="Detailed documentation available at https://noc-msm.github.io/pyIC/.",
    )
    parser.add_argument("--source", "-s", help="path to source grid file")
    parser.add_argument("--destination", "-d", help="path to destination grid file")
    parser.add_argument("--in_data", "-i", help="path to data to be regridded")
    parser.add_argument(
        "--out_path", "-o", nargs="?", const="regridded.nc", help="path to write regridded data to"
    )
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
    grid1_regrid = regrid_data(args.in_data, regridder=regridder)
    grid1_regrid.to_netcdf(args.out_path)


if __name__ == "__main__":
    main()
