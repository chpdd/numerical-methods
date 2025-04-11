import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# Пример правой части ОДУ: y' = y
def f(t, y):
    # Измените выражение справа на нужное.
    return y


def exact_solution(t):
    return np.exp(t)


T = 1.0
y0 = 1.0
h = 0.1


def implicit_euler(f, h, T, y0):
    N = int(T / h)
    t = np.linspace(0, T, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    for n in range(N):
        t_next = t[n + 1]
        g = lambda Y: Y - y[n] - h * f(t_next, Y)
        Y_guess = y[n] + h * f(t[n], y[n])
        y[n + 1] = fsolve(g, Y_guess)[0]
    return t, y


# Решение задачи для двух шагов: h и h/2
t1, y_num1 = implicit_euler(f, h, T, y0)
t2, y_num2 = implicit_euler(f, h / 2, T, y0)

# Если аналитическое решение известно, его можно задать.
# Для y' = y аналитическое решение: y = exp(t)

y_exact1 = exact_solution(t1)
y_exact2 = exact_solution(t2)

# Вычисление ошибок
error1 = np.abs(y_num1 - y_exact1)
error2 = np.abs(y_num2 - y_exact2)
max_error1 = np.max(error1)
max_error2 = np.max(error2)
print(f"Максимальная ошибка для h = {h}: {max_error1:.5e}")
print(f"Максимальная ошибка для h/2 = {h / 2}: {max_error2:.5e}")

# Оценка порядка точности: порядок = log(error1/error2) / log(2)
order = np.log(max_error1 / max_error2) / np.log(2)
print(f"Фактический порядок точности метода: {order:.2f}")

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(t1, y_num1, 'bo-', label=f'Численное решение, h={h}')
plt.plot(t2, y_num2, 'go-', label=f'Численное решение, h={h / 2}')
t_fine = np.linspace(0, T, 200)
plt.plot(t_fine, exact_solution(t_fine), 'r', label='Аналитическое решение')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Сравнение решений (неявный метод Эйлера)')
plt.legend()
plt.grid(True)
plt.show()
