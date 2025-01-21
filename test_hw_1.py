import hw_1
import pytest
import numpy as np


class TestAddTwoNumbers:
    def test_add_two_floats(self):
        assert hw_1.add_two_numbers(1.0, 2.4) == 3.4
        assert hw_1.add_two_numbers(0.0, 0.0) == 0.0
        assert hw_1.add_two_numbers(-1.0, -1.2) == -2.2
        assert hw_1.add_two_numbers(1.0, -1.0) == 0.0

    def test_add_two_ints(self):
        assert hw_1.add_two_numbers(3, 9) == 12
        assert hw_1.add_two_numbers(0, 0) == 0
        assert hw_1.add_two_numbers(3, -9) == -6

    def test_add_mixed_types(self):
        assert hw_1.add_two_numbers(2, 0.2) == 2.2
        assert hw_1.add_two_numbers(0.0, 0) == 0.0
        assert hw_1.add_two_numbers(3.0, -9) == -6.0


class TestFirstKFibonacciNumbers:
    def test_k_equals_0(self):
        assert (hw_1.first_k_fibonacci_numbers(0) == []).all()

    def test_k_equals_1(self):
        assert hw_1.first_k_fibonacci_numbers(1) == [0]

    def test_k_equals_2(self):
        assert (hw_1.first_k_fibonacci_numbers(2) == [0, 1]).all()

    def test_k_equals_5(self):
        assert (hw_1.first_k_fibonacci_numbers(5) == [0, 1, 1, 2, 3]).all()

    def test_k_equals_10(self):
        assert (
            hw_1.first_k_fibonacci_numbers(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        ).all()


class TestReturnLastColumn:
    def test_single_array(self):
        assert hw_1.return_last_column(np.array([[1, 2, 3]])) == [3]

    def test_3x3_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert (hw_1.return_last_column(matrix) == [3, 6, 9]).all()

    def test_4x2_matrix(self):
        matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        assert (hw_1.return_last_column(matrix) == [2, 4, 6, 8]).all()


class TestFindAllSixes:
    def test_no_sixes(self):
        assert hw_1.find_all_sixes([1, 2, 3, 4, 5]) == []

    def test_three_sixes(self):
        assert hw_1.find_all_sixes([1, 6, 2, 13, 6, 4, 6]) == [1, 4, 6]

    def test_all_sixes(self):
        assert hw_1.find_all_sixes([6, 6, 6, 6, 6]) == [0, 1, 2, 3, 4]


class TestMultiplyTwoMatricesUsingForLoops:
    def test_2x2_matrices(self):
        A = np.random.rand(2, 2)
        B = np.random.rand(2, 2)
        proposed_solution = hw_1.multiply_two_matrices_using_for_loops(A, B)
        print(proposed_solution, A @ B)
        assert np.allclose(A @ B, proposed_solution)

    def test_5x5_matrices(self):
        A = np.random.rand(5, 5)
        B = np.random.rand(5, 5)
        proposed_solution = hw_1.multiply_two_matrices_using_for_loops(A, B)
        assert np.allclose(A @ B, proposed_solution)

    def test_wide_to_skinny_matrix(self):
        A = np.random.rand(5, 2)
        B = np.random.rand(2, 3)
        proposed_solution = hw_1.multiply_two_matrices_using_for_loops(A, B)
        assert np.allclose(A @ B, proposed_solution)

    def test_skinny_to_wide_matrix(self):
        A = np.random.rand(3, 8)
        B = np.random.rand(8, 4)
        proposed_solution = hw_1.multiply_two_matrices_using_for_loops(A, B)
        assert np.allclose(A @ B, proposed_solution)

    def test_error_if_A_too_wide(self):
        A = np.random.rand(3, 4)
        B = np.random.rand(3, 5)
        with pytest.raises(ValueError):
            proposed_solution = hw_1.multiply_two_matrices_using_for_loops(A, B)

    def test_error_if_A_too_skinny(self):
        A = np.random.rand(3, 4)
        B = np.random.rand(5, 4)
        with pytest.raises(ValueError):
            proposed_solution = hw_1.multiply_two_matrices_using_for_loops(A, B)
