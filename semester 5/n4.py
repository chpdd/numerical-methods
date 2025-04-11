import numpy as np
import autograd.numpy as anp
from autograd import jacobian
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def plot_system(func, x0_start, x0_end, x1_start, x1_end):
    x0_vals = np.linspace(x0_start, x0_end, 100)
    x1_vals = np.linspace(x1_start, x1_end, 100)
    x0, x1 = np.meshgrid(x0_vals, x1_vals)

    f_values = [func([x0, x1])[i] for i in range(2)]

    plt.figure(figsize=(8, 6))

    contour1 = plt.contour(x0, x1, f_values[0], levels=[0], colors='r')
    contour2 = plt.contour(x0, x1, f_values[1], levels=[0], colors='b')

    plt.title('Графическое решение системы уравнений')
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.grid(True)

    plt.clabel(contour1, inline=True, fontsize=10, fmt='f1 = 0')
    plt.clabel(contour2, inline=True, fontsize=10, fmt='f2 = 0')

    plt.show()


def function_system(x):
    # x0 = [0.5, 0.5]
    f1 = x[0] ** 2 + x[1] ** 2 - 1
    f2 = x[0] - anp.sin(x[1])

    # x0 = [0, 0]
    # f1 = anp.exp(x[0]) + x[1] ** 2 - 1
    # f2 = x[0] ** 2 + x[1] - 1

    # x0 = [0.5, 0.5]
    # f1 = anp.sin(x[0]) + x[1] - 1
    # f2 = x[0] ** 2 + anp.cos(x[1]) - 0.5

    return anp.array([f1, f2])


# Универсальная функция метода Ньютона
def newton_method(func_system, x0, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)
    system_jacobian = jacobian(func_system)
    for i in range(max_iter):
        calc_jacobian = system_jacobian(x)
        calc_system = func_system(x)
        delta_x = np.linalg.solve(calc_jacobian, -calc_system)
        x = x + delta_x
        if np.linalg.norm(delta_x) < tol:
            print(f"Решение найдено за {i + 1} итераций.")
            return x
    print(f"Решение не достигло нужной точности после {max_iter} итераций.")
    return x


def main():
    x0 = np.array([-2.5, -2.5], dtype=float)  # Начальное приближение
    if x0.size == 2:
        plot_system(function_system, -10, 10, -10, 10)
    solution = newton_method(function_system, x0)
    print(f"Решение системы: {solution}")
    print(f"Решение системы с помощью fsolve: {fsolve(function_system, x0)}")


if __name__ == '__main__':
    main()
