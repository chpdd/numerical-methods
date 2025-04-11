import numpy as np
from functions import f, g

def aitken_process(x0, x1, x2):
    if x1 == x0 or x2 == x1:
        return None
    return x2 - ((x2 - x1) ** 2) / (x2 - 2 * x1 + x0)

def simple_iteration_method(a, b, tol=1e-6, max_iter=1000):
    x0 = a
    x1 = g(x0)

    roots = []

    for i in range(max_iter):
        if abs(f(x1)) < tol:
            roots.append(x1)
            break

        x2 = g(x1)

        x_aitken = aitken_process(x0, x1, x2)

        if x_aitken is not None:
            x0, x1 = x1, x_aitken
        else:
            x0, x1 = x1, x2

        if abs(x1 - x0) < tol:
            roots.append(x1)
            break

    return roots

a = 1
b = 3
roots = simple_iteration_method(a, b)

with open('output.txt', 'w') as file:
    file.write("Current roots: {}\n".format(np.array(roots)))
