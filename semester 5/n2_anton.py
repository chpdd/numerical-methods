import numpy as np
import matplotlib.pyplot as plt

# Определяем нечётный полином
def odd_polynomial(x):
    return 5*x**5 - 2*x

# Генерируем точки для интерполяции
x_data = np.linspace(-3, 3, 10)  # 10 точек от -3 до 3
y_data = odd_polynomial(x_data)

# Создаем тригонометрическую интерполяцию
def trig_interpolation(x, n):
    a0 = (1 / (2 * n)) * np.sum(y_data)
    a = np.zeros(n)
    b = np.zeros(n)

    for k in range(1, n):
        a[k] = (1 / n) * np.sum(y_data * np.cos(2 * np.pi * k * x_data / n))
        b[k] = (1 / n) * np.sum(y_data * np.sin(2 * np.pi * k * x_data / n))

    return a0 + np.sum([a[k] * np.cos(2 * np.pi * k * x / n) + b[k] * np.sin(2 * np.pi * k * x / n) for k in range(n)], axis=0)

# Интерполируемые точки
x_interp = np.linspace(-3, 3, 100)
y_interp = trig_interpolation(x_interp, len(x_data))

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'ro', label='Исходные точки')
plt.plot(x_interp, y_interp, 'b-', label='Тригонометрическая интерполяция')
plt.plot(x_interp, odd_polynomial(x_interp), 'g--', label='Нечётный полином')
plt.title('Тригонометрическая интерполяция нечётного полинома')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()
