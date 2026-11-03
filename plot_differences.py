import numpy as np
import matplotlib.pyplot as plt
import sys


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python plot_difference.py file1.dat file2.dat")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    M = 128
    N = 128

    data1 = load_data(file1, M, N)
    data2 = load_data(file2, M, N)

    # Compute difference
    diff = data2 - data1

    plt.figure(figsize=(6, 5))
    im = plt.imshow(diff, origin='lower', cmap='bwr', interpolation='nearest')
    plt.colorbar(im, label="Difference")
    plt.title(f"Difference between {file2} and {file1}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()

    # Save figure to a file
    plt.savefig("difference_plot.png", dpi=150)
    print("Saved difference plot to difference_plot.png")


def load_data(filename:str, M:int, N:int) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.float64)
    if data.size != M * N:
        raise ValueError(f"File {filename} does not contain M*N={M*N} entries")
    return data.reshape((M, N))

if __name__ == "__main__":
    main()
