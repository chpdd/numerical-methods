import numpy as np


def read_file(file_name):
    with open(file_name, 'r') as input_file:
        rows = input_file.readlines()
    n = int(rows[0].strip())
    A = []
    for row in rows[1: n + 1]:
        A.append(list(map(int, row.strip().split())))
    b = list(map(int, rows[n + 1].strip().split()))
    return A, b


def qr_decomposition(A):
    """Функция для QR-разложения матрицы A."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def solve_slae(A, b):
    """Функция для решения СЛАУ Ax = b методом QR."""
    # QR-разложение
    Q, R = qr_decomposition(A)

    # Обратное преобразование
    y = np.dot(Q.T, b)

    # Решение Rx = y
    x = np.linalg.solve(R, y)

    return x


def main():
    file_number = int(input("Введите номер файла: "))
    file_name = f'input_n1_{file_number}.txt'
    A, b = read_file(file_name)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    result = solve_slae(A, b)
    print("Решение СЛАУ: ", result, sep="\n")


if __name__ == '__main__':
    main()
