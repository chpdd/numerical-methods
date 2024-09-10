import numpy as np

a = np.array([[1.1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([10, 11, 12])
b = b.reshape(1, -1)
print(b)
# b = b.reshape(-1, 1)
a = np.hstack((a, b.T))
print(a[0][0].dtype)