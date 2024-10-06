import numpy as np
import matplotlib.pyplot as plt


def read_file(file_name):
    with open(file_name, 'r') as input_file:
        rows = input_file.readlines()
    x_data = np.array(list(map(float, rows[0].strip().split())))
    y_data = np.array(list(map(float, rows[1].strip().split())))
    return x_data, y_data


def aitken_interpolation(x_data, y_data, x):
    n = len(x_data)
    P = np.zeros((n, n))
    P[:, 0] = y_data

    for j in range(1, n):
        for i in range(n - j):
            P[i, j] = ((x - x_data[i + j]) * P[i, j - 1] - (x - x_data[i]) * P[i + 1, j - 1]) / (
                    x_data[i] - x_data[i + j])
    # print(x, P, sep="\n")
    return P[0, n - 1]


def main():
    file_number = int(input('Введите номер примера: '))
    file_name = f'input_n2_{file_number}.txt'
    x_data, y_data = read_file(file_name)

    x_min = min(x_data)
    x_max = max(x_data)

    x_values = np.linspace(min(x_data), max(x_data), int(x_max - x_min) * 10)
    y_values = [aitken_interpolation(x_data, y_data, x) for x in x_values]

    plt.plot(x_data, y_data, 'o', label='Табличные данные', color='red')
    plt.plot(x_values, y_values, label='Интерполяция', color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Интерполяция табличной функции по Схеме Эйткена')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
