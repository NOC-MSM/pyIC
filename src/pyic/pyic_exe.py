import sys

from src.pyic.grid import GRID
from src.pyic.regrid import make_regridder

file1 = sys.argv[1]
file2 = sys.argv[2]


def main():
    print("pyIC")
    grid1 = GRID(file1)
    grid2 = GRID(file2)
    regridder = make_regridder(grid1, grid2, save_weights="weights.nc")
    regridder.to_netcdf("regridded.nc")


if __name__ == "__main__":
    main()
