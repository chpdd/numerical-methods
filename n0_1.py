from decimal import Decimal as dec
import numpy as np


class NoSolutionException(Exception):
    def __init__(self, message):
        super().__init__(message)


def get_matrix_from_file(file_name: str):
    result = []
    with open(file_name, "r") as input_file:
        lines = input_file.readlines()
        n = int(lines[0].strip())
        for line in lines[1: n + 1]:
            result.append(list(map(float, line.strip().split())))
        b = np.array(list(map(float, lines[n + 1].strip().split()))).reshape(-1, 1)
        result = np.hstack([np.array(result), b])

    return result


def triangularize(input_matrix):
    result = input_matrix
    n = result.shape[0]

    for diag_i in range(0, n - 1):
        if result[diag_i, diag_i] == 0:
            non_zero_row = np.where(result[diag_i + 1:, diag_i] != 0)
            if non_zero_row[0].size == 0:
                raise NoSolutionException("СЛАУ имеет не единственное решение")
            non_zero_row_index = non_zero_row[0][0] + (diag_i + 1)
            result[[diag_i, non_zero_row_index]] = result[[non_zero_row_index, diag_i]]
        for under_i in range(diag_i + 1, n):
            if result[under_i, diag_i] != 0:
                factor = result[under_i, diag_i] / result[diag_i, diag_i]
                result[under_i] = result[under_i] - (factor * result[diag_i])

    return result


def solve_system(input_matrix):
    n = input_matrix.shape[0]
    result = np.array([1.0] * n)

    for i in range(n - 1, -1, -1):
        result[i] = ((input_matrix[i, -1] - np.dot(input_matrix[i, i + 1: n], result[i + 1: n]))
                     / input_matrix[i, i])
    return result


def main():
    try:
        matrix = get_matrix_from_file("input_n0.txt")
        matrix = triangularize(matrix)
        solution = solve_system(matrix)
        for i in range(0, len(solution)):
            print(f"x{i + 1} = {solution[i]}")
    except NoSolutionException as e:
        print(e)


if __name__ == "__main__":
    main()
# 5
# 2 3 1 4 5
# 1 4 2 3 6
# 3 2 5 6 1
# 4 1 6 5 2
# 5 6 4 1 3
# 10 12 14 16 18

# 3
# 1 2 0
# 4 0 5
# 6 9 0
# 10 4 6
#
# 3
# 1 2 3
# 4 5 6
# 7 8 9
# 3 6 9
