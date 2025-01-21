# from hw_1 import plot_matrix_multiplication_timings
from hw_1_solutions import plot_matrix_multiplication_timings

if __name__ == "__main__":
    plot_matrix_multiplication_timings(
        [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], plot_log_scale=True
    )
