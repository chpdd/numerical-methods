import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def backward_euler(f, y0, x, h):
    # x - список [левая граница, правая граница]
    # h - шаг
    # f - правая часть ОДУ

    N_t = int(round((x[1] - x[0]) / h))  # Количество шагов по времени
    f_ = lambda t, y: np.asarray(f(t, y))  # Преобразуем f в форму, которая будет возвращать numpy-массив
    t = np.linspace(x[0], x[1], N_t + 1)  # Массив всех временных точек, от x[0] до x[1] с шагом h
    y = np.zeros((N_t + 1, len(y0)))  # Массив для значений решения, y[0] — начальное условие
    y[0] = y0  # Устанавливаем начальное условие

    def Phi(z, t, v):
        return z - h * f_(t, z) - v  # Нелинейное уравнение для поиска следующего значения

    for n in range(N_t):
        # Для каждого шага решаем нелинейное уравнение
        y[n + 1] = optimize.fsolve(Phi, y[n], args=(t[n], y[n]))  # Используем fsolve для нахождения корня

    return y, t


# Пример правой части ОДУ: y' = -y + sin(t)
def f(t, y):
    return -y + np.sin(t)


# Начальное условие
y0 = 1  # y(0) = 1

# Интервал [левая граница, правая граница] и шаг h
x = [0, 10]  # Интервал от 0 до 10
h = 0.1  # Шаг

# Решение задачи методом Эйлера
y, t = backward_euler(f, y0, x, h)

# Построим график
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Численное решение методом Эйлера (неявный)')
plt.grid()
plt.show()
