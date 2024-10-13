import numpy as np


def matrixFromFile(filename):
    with open(filename, 'r') as f:
        matrix = [list(map(float, line.split())) for line in f]
    return np.array(matrix)


def matrixToFile(matrix, filename):
    with open(filename, 'w') as f:
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')


def luDecomp(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # сначала Upper, потом Lower

    for i in range(n):
        # нахождение верхней
        for j in range(i, n):
            U[i][j] = matrix[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]

        # нахождение нижней
        for j in range(i, n):
            if i == j:
                L[i][i] = 1  # устанавливаем единицы по диагонали
            else:
                L[j][i] = matrix[j][i]
                for k in range(i):
                    L[j][i] -= L[j][k] * U[k][i]
                L[j][i] /= U[i][i]

    return L, U


def forwardSub(L, b):
    n = len(L)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]  # вектор свободных членов - нулевой
        for j in range(i):
            y[i] -= L[i][j] * y[j]
    return y


def backwardSub(U, y):
    n = len(U)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    return x


def invertMatrix(matrix):
    n = len(matrix)
    L, U = luDecomp(matrix)

    invMatrix = np.zeros((n, n))

    for i in range(n):
        # базисный вектор единичной длины
        e_i = np.zeros(n)
        e_i[i] = 1
        y = forwardSub(L, e_i)
        invMatrix[:, i] = backwardSub(U, y)

    return invMatrix


def main(input_file, output_file):
    matrix = matrixFromFile(input_file)
    invMatrix = invertMatrix(matrix)

    matrixToFile(invMatrix, output_file)


input_file = 'input.txt'
output_file = 'output.txt'
main(input_file, output_file)
