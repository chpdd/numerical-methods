import numpy as np
import matplotlib.pyplot as plt


class MaxIterationsError(Exception):
    pass


class DiscriminantError(Exception):
    pass


def plot_parabola(A, B, C, x_min, x_max, color='r', label=None):
    x = np.linspace(x_min, x_max, 500)
    y = A * x ** 2 + B * x + C
    plt.plot(x, y, color, label=label)


def parabolic_interpolation(a, b, tol=1e-7, max_iter=100):
    x0 = a
    x1 = (a + b) / 2
    x2 = b

    for i in range(max_iter):
        f0, f1, f2 = f(x0), f(x1), f(x2)

        denominator = (x0 - x1) * (x0 - x2) * (x1 - x2)
        A = (x2 * (f1 - f0) + x1 * (f0 - f2) + x0 * (f2 - f1)) / denominator
        B = (x2 ** 2 * (f0 - f1) + x1 ** 2 * (f2 - f0) + x0 ** 2 * (f1 - f2)) / denominator
        C = (x1 * x2 * (x1 - x2) * f0 + x2 * x0 * (x2 - x0) * f1 + x0 * x1 * (x0 - x1) * f2) / denominator
        plot_parabola(A, B, C, a, b)

        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0:
            raise DiscriminantError("Дискриминант отрицательный. Нет действительных корней.")

        sqrt_discriminant = np.sqrt(discriminant)
        x_new1 = (-B + sqrt_discriminant) / (2 * A)
        x_new2 = (-B - sqrt_discriminant) / (2 * A)

        x_new = x_new1 if abs(x_new1 - x1) < abs(x_new2 - x1) else x_new2

        if abs(f(x_new)) < tol:
            plt.scatter(x_new, f(x_new), color='g', zorder=5)
            return x_new, i + 1

        if abs(f(x_new) - f(x0)) < tol:
            x0 = x_new
        elif abs(f(x_new) - f(x1)) < tol:
            x1 = x_new
        else:
            x2 = x_new

    raise MaxIterationsError("Метод не сошёлся после максимального числа итераций.")


def f(x):
    # 1:4
    return x ** 3 - 6 * x ** 2 + 4 * x + 12
    # -3:2
    # return x ** 2 - 4
    # 2:6
    # return np.log(x) - 1


def main():
    a, b = 1, 4
    # a, b = -3, 2
    # a, b = 2, 6

    x_vals = np.linspace(a, b, 500)
    y_vals = f(x_vals)
    plt.plot(x_vals, y_vals, label='Изначальная функция', color='blue')
    try:
        root, iters_n = parabolic_interpolation(a, b)
        if a <= root <= b:
            plt.title(f"Найденный корень на отрезке [{a}, {b}]: {root}")
            plt.axhline(0, color='black', linewidth=0.5)
            plt.axvline(root, color='g', linestyle='--', label=f'Корень: {root}')
            plt.legend()
            plt.grid(True)
            plt.show()

            print(f"Количество итераций: {iters_n}")
            print(f"Найденный корень на отрезке [{a}, {b}]: {root}")
        else:
            print("Найденный корень находится за отрезком")
    except (MaxIterationsError, DiscriminantError) as error:
        print(error)


if __name__ == '__main__':
    main()
