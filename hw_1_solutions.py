"""
Homework 1: Intro to Programming"""

import numpy as np
import matplotlib.pyplot as plt
import time


def add_two_numbers(a: float, b: float) -> float:
    """
    Evaluates the sum of two numbers and returns the result.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The sum of a and b.

    Example:
        add_two_numbers(1.0, 2.0) -> 3.0

    """
    return a + b


def first_k_fibonacci_numbers(k: int) -> list:
    """
    Returns the first k Fibonacci numbers.

    Args:
        k (int): The number of Fibonacci numbers to return.

    Returns:
        np.ndarray: Numpy array containing the first k Fibonacci numbers.

    Example:
        first_k_fibonacci_numbers(5) -> [0, 1, 1, 2, 3]
        first_k_fibonacci_numbers(1) -> [0]
    """
    output = [0, 1]

    for i in range(2, k):
        last_number = output[-1]
        second_last_number = output[-2]
        output.append(last_number + second_last_number)

    output = np.array(output)
    return output[:k]


def find_all_sixes(x: np.ndarray) -> np.ndarray:
    """
    Returns an array containing the indices of all the 6s in an array of integers.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: Numpy array containing the indices of all the 6s in the input array.

    Example:
        find_all_sixes(np.array([1, 6, 2, 13, 6, 4, 6])) -> np.array([1, 4, 6])
        find_all_sixes(np.array([1, 2, 3, 4, 5])) -> np.array([])
        find_all_sixes(np.array([6, 6, 6, 6, 6])) -> np.array([0, 1, 2, 3, 4])
    """
    # Using a python magic function:
    # return np.where(x == 6)[0]

    # Using a for loop:
    indices = []
    for i in range(len(x)):
        if x[i] == 6:
            indices.append(i)

    return indices


def return_last_column(A: np.ndarray) -> np.ndarray:
    """
    Returns the last column of a matrix.

    Args:
        A (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The last column of the input matrix.

    Example:
        return_the_last_column(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) -> np.array([3, 6, 9])
    """

    return A[:, -1]


def multiply_two_matrices_using_for_loops(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiplies two matrices by looping over rows and columns and taking the
    the dot product as appropriate.

    Args:
        A: np.ndarray
        B: np.ndarray

    Returns:
        np.ndarray: The product of A and B.
        If matrices cannot be multiplied, raises a ValueError

    Example:
        multiply_two_matrices_using_for_loops(np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])) -> np.array([[7, 10], [15, 22]])
    """
    # Initialize a matrix to store the result, should be of size (A_rows, B_cols)
    rows_in_A = A.shape[0]
    cols_in_B = B.shape[1]
    result_matrix = np.zeros((rows_in_A, cols_in_B))

    # Check if the matrices can be multiplied, and raise a ValueError error if they cannot
    # Errors can be raised using the `raise` keyword:
    # raise ValueError("Matrices cannot be multiplied")
    matrices_cannot_be_multiplied = A.shape[1] != B.shape[0]
    if matrices_cannot_be_multiplied:
        raise ValueError("Matrices cannot be multiplied")

    # Loop over rows of A, then columns of B, and add to the right entry of the result matrix
    for i in range(rows_in_A):
        for j in range(cols_in_B):
            result_matrix[i, j] = np.dot(A[i, :], B[:, j])

    return result_matrix


def multiply_two_matrices_using_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiplies two matrices using numpy and returns the result.

    Args:
        A (np.ndarray): The first matrix.
        B (np.ndarray): The second matrix.

    Returns:
        np.ndarray: The product of A and B.
    """
    return A @ B  # I already implemented this for you!


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
