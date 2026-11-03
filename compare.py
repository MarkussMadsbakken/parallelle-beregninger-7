import numpy as np
import sys


class bcolors:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def load_data(filename: str, M: int, N: int) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.float64)

    if data.size != M * N:
        raise ValueError(
            f"File {filename} does not contain M*N={M*N} entries "
            f"(found {data.size})"
        )

    return data.reshape((M, N))


def main() -> None:
    error_margin = 1e-4

    if len(sys.argv) != 3:
        print("Usage: python compare.py file1.dat file2.dat")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    M = 128
    N = 128

    data1 = load_data(file1, M, N)
    data2 = load_data(file2, M, N)

    diff = np.abs(data2 - data1)
    max_diff = np.max(diff)

    if max_diff > error_margin:
        print(
            f"\n{bcolors.FAIL}"
            f"Data files {file1} and {file2} differ by more than {error_margin}\n"
            f"Maximum difference: {max_diff}"
            f"{bcolors.ENDC}\n"
        )
        sys.exit(1)

    else:
        print(
            f"\n{bcolors.OKGREEN}"
            f"Data files {file1} and {file2} are identical within the margin of {error_margin}\n"
            f"Maximum difference: {max_diff}"
            f"{bcolors.ENDC}\n"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
    