import numpy as np
import time
import matplotlib.pyplot as plt
from hw_1 import (
    multiply_two_matrices_using_for_loops,
    multiply_two_matrices_using_numpy,
)


def plot_matrix_multiplication_timings(matrix_sizes: np.ndarray, plot_log_scale):
    """
    Plots the time taken for matrix multiplication using for loops and numpy.

    Args:
        matrix_sizes (np.ndarray): An array containing the matrix sizes at
            which to measure timings
    """
    for_loop_timings = []
    numpy_timings = []
    for matrix_size in matrix_sizes:
        A = np.random.rand(matrix_size, matrix_size)
        B = np.random.rand(matrix_size, matrix_size)

        # Time taken for matrix multiplication using for loops
        start_time = time.time()
        multiply_two_matrices_using_for_loops(A, B)
        time_for_loops = time.time() - start_time
        for_loop_timings.append(time_for_loops)

        # Time taken for matrix multiplication using numpy
        start_time = time.time()
        multiply_two_matrices_using_numpy(A, B)
        time_numpy = time.time() - start_time
        numpy_timings.append(time_numpy)

    plt.plot(matrix_sizes, for_loop_timings, color="r", marker="s", label="For Loops")
    plt.plot(matrix_sizes, numpy_timings, color="b", marker="o", label="Numpy")
    plt.legend()
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (s)")

    if plot_log_scale:
        plt.yscale("log")
        plt.xscale("log")

    plt.show()


if __name__ == "__main__":  # This code only runs if you run this script directly
    plot_matrix_multiplication_timings(
        [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], plot_log_scale=True
    )
