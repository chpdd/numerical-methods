from decimal import Decimal as dec

class NoSolutionException(Exception):
    def __init__(self, message):
        super().__init__(message)


def get_matrix_from_file(file_name: str):
    result = []
    with open(file_name, "r") as input_file:
        lines = input_file.readlines()
        n = int(lines[0].strip())
        b = list(map(dec, lines[-1].strip().split()))
        for line in lines[1: n + 1]:
            result.append(list(map(dec, line.strip().split())) + [b[0]])
            del b[0]
    return result, n


def triangularize(input_matrix, n: int):
    result = input_matrix
    for i in range(0, n):
        if result[i][i] == 0:
            for i2 in range(i + 1, n):
                if result[i2][i] != 0:
                    result[i], result[i2] = result[i2], result[i]
                    break
            if result[i][i] == 0:
                raise NoSolutionException("СЛАУ не имеет единственного решения")
        for i2 in range(i + 1, n):
            if result[i2][i] == 0:
                continue
            k = result[i2][i] / result[i][i]
            for j in range(i - 1, n + 1):
                if j == -1:
                    continue
                # print(i, i2, j)
                result[i2][j] -= k * result[i][j]
                # print(k)
                # print(*result, sep="\n", end="\n\n")
    return result


def solve_system(input_matrix, n: int):
    result = [dec('1')] * n
    for i in range(n - 1, -1, -1):
        line_sum = 0
        for j in range(i + 1, n):
            line_sum += input_matrix[i][j] * result[j]
        result[i] = (input_matrix[i][-1] - line_sum) / input_matrix[i][i]
    return result


def main():
    try:
        matrix, n = get_matrix_from_file("input_n0.txt")
        matrix = triangularize(matrix, n)
        solution = solve_system(matrix, n)
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
