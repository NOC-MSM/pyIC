import argparse

from pyic.grid import GRID
from pyic.regrid import make_regridder


def main():
    parser = argparse.ArgumentParser(
        prog="pyic",
        description="pyIC: Generate initial conditions for regional NEMO configurations out of the box.",
        epilog="Detailed documentation available at https://noc-msm.github.io/pyIC/.",
    )
    parser.add_argument("source", help="path to source data/grid file")
    parser.add_argument("destination", help="path to destination grid file")
    args = parser.parse_args()
    grid1 = GRID(data_filename=args.source)
    grid2 = GRID(data_filename=args.destination)
    regridder = make_regridder(grid1, grid2, save_weights="weights.nc")
    regridder.to_netcdf("regridded.nc")


if __name__ == "__main__":
    main()
