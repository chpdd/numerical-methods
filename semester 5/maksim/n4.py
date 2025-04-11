import numpy as np


# Определим систему нелинейных уравнений
def f(x):
    # Система: f1(x) = x1^2 + x2^2 - 4, f2(x) = x1*x2 - 1
    f1 = x[0] ** 2 + x[1] ** 2 - 4
    f2 = x[0] * x[1] - 1
    return np.array([f1, f2])


# Якобиан системы
def jacobian(x):
    # Частные производные для f1 и f2 по x1 и x2
    df1_dx1 = 2 * x[0]
    df1_dx2 = 2 * x[1]
    df2_dx1 = x[1]
    df2_dx2 = x[0]

    return np.array([[df1_dx1, df1_dx2], [df2_dx1, df2_dx2]])


# Метод Ньютона для решения системы
def newton_method(f, jacobian, x0, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)  # Начальное приближение

    for _ in range(max_iter):
        # Вычисляем значения функций и якобиан
        F = f(x)
        J = jacobian(x)

        # Решаем систему J * delta_x = -F
        delta_x = np.linalg.solve(J, -F)

        # Обновляем решение
        x = x + delta_x

        # Проверяем на сходимость
        if np.linalg.norm(delta_x, ord=2) < tol:
            return x

    # Если решение не найдено за max_iter итераций
    return x


# Чтение начальных значений из файла
def read_initial_values(filename):
    with open(filename, 'r') as file:
        # Читаем значения из первой строки, разделённые пробелом или новой строкой
        values = file.readline().split()
        return [float(values[0]), float(values[1])]


# Запись результата в файл
def write_result(filename, result):
    with open(filename, 'w') as file:
        file.write(f"Решение системы: x1 = {result[0]}, x2 = {result[1]}\n")


# Основная программа
def main():
    # Чтение начальных значений из файла input.txt
    x0 = read_initial_values('input.txt')

    # Применяем метод Ньютона
    solution = newton_method(f, jacobian, x0)

    # Записываем результат в output.txt
    write_result('output.txt', solution)
    print(f"Решение записано в output.txt")


# Запуск программы
if __name__ == "__main__":
    main()