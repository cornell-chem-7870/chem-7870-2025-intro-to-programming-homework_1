import hw_1
import pytest
import numpy as np


def test_add_two_floats():
    assert hw_1.add_two_numbers(1.0, 2.4) == 3.4
    assert hw_1.add_two_numbers(0.0, 0.0) == 0.0
    assert hw_1.add_two_numbers(-1.0, -1.2) == -2.2
    assert hw_1.add_two_numbers(1.0, -1.0) == 0.0


def test_add_two_ints():
    assert hw_1.add_two_numbers(3, 9) == 12
    assert hw_1.add_two_numbers(0, 0) == 0
    assert hw_1.add_two_numbers(3, -9) == -6


def test_add_mixed_types():
    assert hw_1.add_two_numbers(2, 0.2) == 2.2
    assert hw_1.add_two_numbers(0.0, 0) == 0.0
    assert hw_1.add_two_numbers(3.0, -9) == -6.0


def test_k_equals_0():
    assert np.allclose(hw_1.first_k_fibonacci_numbers(0), np.array([]))


def test_k_equals_1():
    assert np.allclose(hw_1.first_k_fibonacci_numbers(1), np.array([0]))


def test_k_equals_2():
    assert np.allclose(hw_1.first_k_fibonacci_numbers(2), np.array([0, 1]))


def test_k_equals_5():
    assert np.allclose(hw_1.first_k_fibonacci_numbers(5) , np.array([0, 1, 1, 2, 3]))


def test_k_equals_10():
    np.allclose(
        hw_1.first_k_fibonacci_numbers(10),
        np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    )


def test_single_array():
    assert np.allclose(hw_1.return_last_column(np.array([[1, 2, 3]])), [3])


def test_3x3_matrix():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert np.allclose(hw_1.return_last_column(matrix), [3, 6, 9])


def test_4x2_matrix():
    matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert np.allclose(hw_1.return_last_column(matrix), [2, 4, 6, 8])


def test_no_sixes():
    assert np.allclose(hw_1.find_all_sixes([1, 2, 3, 4, 5]), [])


def test_three_sixes():
    assert np.allclose(hw_1.find_all_sixes([1, 6, 2, 13, 6, 4, 6]),  [1, 4, 6])


def test_all_sixes():
    assert np.allclose(hw_1.find_all_sixes([6, 6, 6, 6, 6]),  [0, 1, 2, 3, 4])


def test_2x2_matrices():
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 2)
    proposed_solution = hw_1.multiply_two_matrices_using_for_loops(A, B)
    print(proposed_solution, A @ B)
    assert np.allclose(A @ B, proposed_solution)


def test_5x5_matrices():
    A = np.random.rand(5, 5)
    B = np.random.rand(5, 5)
    proposed_solution = hw_1.multiply_two_matrices_using_for_loops(A, B)
    assert np.allclose(A @ B, proposed_solution)


def test_wide_to_skinny_matrix():
    A = np.random.rand(5, 2)
    B = np.random.rand(2, 3)
    proposed_solution = hw_1.multiply_two_matrices_using_for_loops(A, B)
    assert np.allclose(A @ B, proposed_solution)


def test_skinny_to_wide_matrix():
    A = np.random.rand(3, 8)
    B = np.random.rand(8, 4)
    proposed_solution = hw_1.multiply_two_matrices_using_for_loops(A, B)
    assert np.allclose(A @ B, proposed_solution)


def test_error_if_A_too_wide():
    A = np.random.rand(3, 4)
    B = np.random.rand(3, 5)
    with pytest.raises(ValueError):
        proposed_solution = hw_1.multiply_two_matrices_using_for_loops(A, B)


def test_error_if_A_too_skinny():
    A = np.random.rand(3, 4)
    B = np.random.rand(5, 4)
    with pytest.raises(ValueError):
        proposed_solution = hw_1.multiply_two_matrices_using_for_loops(A, B)
