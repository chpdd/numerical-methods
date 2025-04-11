# 2б
import numpy as np  # Импортируем библиотеку для работы с матрицами и векторами


def inverse_power_method(A, tol=1e-6, max_iter=1000):
    """
    Реализация обратного степенного метода для нахождения наименьшего по модулю собственного значения и вектора.

    Параметры:
    A - квадратная матрица (numpy array)
    tol - точность остановки алгоритма (по умолчанию 1e-6)
    max_iter - максимальное количество итераций (по умолчанию 1000)

    Возвращает:
    lambda_min - приближенное наименьшее собственное значение
    x_min - соответствующий собственный вектор
    """
    n = A.shape[0]  # Определяем размерность матрицы
    x = np.random.rand(n)  # Создаем случайный начальный вектор размерности n
    x = x / np.linalg.norm(x)  # Нормируем вектор, чтобы избежать числовой нестабильности

    A_inv = np.linalg.inv(A)  # Вычисляем обратную матрицу A

    lambda_old = 0  # Инициализируем переменную для хранения предыдущего значения собственного числа
    for _ in range(max_iter):  # Запускаем итерационный процесс
        x_new = A_inv @ x  # Умножаем текущий вектор на обратную матрицу A
        x_new = x_new / np.linalg.norm(x_new)  # Нормируем полученный вектор

        lambda_new = x_new @ A @ x_new  # Вычисляем приближенное собственное значение

        if np.abs(lambda_new - lambda_old) < tol:  # Проверяем условие сходимости
            break  # Если разница между итерациями меньше tol, останавливаем процесс

        x = x_new  # Обновляем вектор
        lambda_old = lambda_new  # Обновляем собственное значение

    return lambda_new, x_new  # Возвращаем найденное наименьшее собственное значение и вектор


import numpy as np


def read_matrix_from_file(filename):
    try:
        with open(filename, "r", encoding="UTF-8") as file:
            matrix = [list(map(int, line.split())) for line in file]
        return np.array(matrix)
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None

def write_str_to_file(filename, data):
    try:
        with open(filename, "w", encoding="UTF-8") as file:
            file.write(data)
    except Exception as e:
        print(e)

A = read_matrix_from_file("input.txt")

if A is None:
    print("Ошибка: не удалось загрузить матрицу.")
elif A.shape[0] != A.shape[1]:
    print("Ошибка: матрица должна быть квадратной.")
else:
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("Ошибка: матрица вырождена (det(A) = 0), нельзя найти обратную.")

lambda_min, x_min = inverse_power_method(A)
result = f"Наименьшее собственное значение: {lambda_min}\n"
result += f"Соответствующий собственный вектор: {x_min}"

write_str_to_file("output.txt", result)
