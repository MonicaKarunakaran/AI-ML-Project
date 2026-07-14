import numpy as np
print("===== 1D Array =====")
a = np.array([1, 2, 3, 4, 5])
print(a)
print("Shape:", a.shape)
print("\n===== 2D Array =====")
b = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(b)
print("Shape:", b.shape)
print("\n===== 3D Array =====")
c = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(c)
print("Shape:", c.shape)



print("\n===== Broadcasting =====")
arr = np.array([1, 2, 3])
result = arr + 10
print("Original:", arr)
print("After Broadcasting:", result)



print("\n===== Vectorized Operations =====")
x = np.array([10, 20, 30])
y = np.array([1, 2, 3])
print("Addition:", x + y)
print("Subtraction:", x - y)
print("Multiplication:", x * y)
print("Division:", x / y)


print("\n===== Matrix Multiplication =====")
A = np.array([
    [1, 2],
    [3, 4]
])
B = np.array([
    [5, 6],
    [7, 8]
])
print(np.matmul(A, B))



import pandas as pd
print("\n===== CSV Statistics =====")
df = pd.read_csv("students.csv")
print(df)
print("\nMean")
print(df.mean(numeric_only=True))
print("\nStandard Deviation")
print(df.std(numeric_only=True))
print("\nCorrelation Matrix")
print(df.corr(numeric_only=True))


