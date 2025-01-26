"""
Homework 1: Intro to Programming"""

import numpy as np


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
    raise NotImplementedError


def first_k_fibonacci_numbers(k: int) -> np.ndarray:
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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

    # Check if the matrices can be multiplied, and raise a ValueError error if they cannot
    # Errors can be raised using the `raise` keyword:
    # raise ValueError("Matrices cannot be multiplied")

    # Loop over rows of A, then columns of B, and add to the right entry of the result matrix
    raise NotImplementedError


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
