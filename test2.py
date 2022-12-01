import numpy as np

x = np.random.rand(10, 3)
w = np.random.rand(10, 5)
b = np.random.rand(5, 1)
y = np.dot(w.T, x) + b

print(np.dot(w.T, x))
print(b)
print(y)
