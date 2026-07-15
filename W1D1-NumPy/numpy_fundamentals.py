"""W1D1: NumPy Fundamentals.

Demonstrates core array operations, vectorization, broadcasting, 
matrix multiplication, and dataset statistics using NumPy and Pandas.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# 2️⃣ Use pathlib.Path for file handling
CSV_FILE = Path("students.csv")
OFFSET = 10  # 8️⃣ Avoid hard-coded magic numbers


# 1️⃣ Structure the script into functions & 5️⃣ Add type hints and docstrings
def demo_arrays() -> None:
    """Create 1-D, 2-D, and 3-D arrays and assert their dimensions."""
    print("\n===== 1D Array =====")
    a = np.array([1, 2, 3, 4, 5])
    print(a)
    print(f"Shape: {a.shape}")
    # 9️⃣ Add sanity checks (assertions)
    assert a.ndim == 1 and a.shape == (5,), "Failed 1D array sanity check"

    print("\n===== 2D Array =====")
    b = np.array([[1, 2, 3], [4, 5, 6]])
    print(b)
    print(f"Shape: {b.shape}")
    assert b.ndim == 2 and b.shape == (2, 3), "Failed 2D array sanity check"

    print("\n===== 3D Array =====")
    c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(c)
    print(f"Shape: {c.shape}")
    assert c.ndim == 3 and c.shape == (2, 2, 2), "Failed 3D array sanity check"


def demo_broadcasting() -> None:
    """Demonstrate broadcasting a 1D vector to a 2D matrix."""
    print("\n===== Broadcasting =====")
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    vector = np.array([OFFSET, OFFSET, OFFSET])
    
    result = matrix + vector
    print(f"Original:\n{matrix}")
    print(f"After Broadcasting (Adding {OFFSET}):\n{result}")


def demo_vectorised_ops() -> None:
    """Show vectorized basic arithmetic operations."""
    print("\n===== Vectorized Operations =====")
    x = np.array([10, 20, 30])
    y = np.array([1, 2, 3])
    
    print(f"Addition: {x + y}")
    print(f"Subtraction: {x - y}")
    print(f"Multiplication: {x * y}")
    print(f"Division: {x / y}")


def demo_matmul() -> None:
    """Perform matrix multiplication using the @ operator."""
    print("\n===== Matrix Multiplication =====")
    mat_a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    mat_b = np.array([[7, 8], [9, 10], [11, 12]])  # (3, 2)
    
    result = mat_a @ mat_b  # Equivalent to np.matmul(mat_a, mat_b)
    print(result)
    assert result.shape == (2, 2), "Failed matrix multiplication shape check"


def csv_statistics(csv_path: Path) -> None:
    """Load statistics using Pandas and pure NumPy side-by-side."""
    print("\n===== CSV Statistics =====")
    
    # 3️⃣ Basic error handling
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"❌ {csv_path.name} not found! Place it in the same directory as the script."
        )
        
    # Pandas Version (Side-by-Side)
    df = pd.read_csv(csv_path)
    print("--- Pandas Analysis ---")
    print(df.head())
    print("\nMean (Pandas):")
    print(df[["Math", "Science", "English"]].mean())
    
    # 4️⃣ Pure NumPy Focus (Using ddof=1 for Sample Standard Deviation)
    print("\n--- Pure NumPy Analysis ---")
    # Load columns 1, 2, 3 (Math, Science, English) bypassing names (col 0)
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, usecols=(1, 2, 3))
    
    print(f"Mean (NumPy): {data.mean(axis=0)}")
    print(f"Std  (NumPy, ddof=1): {data.std(axis=0, ddof=1)}")
    print(f"Corr (NumPy):\n{np.corrcoef(data, rowvar=False)}")


# 6️⃣ Use the if __name__ == "__main__" guard
if __name__ == "__main__":
    demo_arrays()
    demo_broadcasting()
    demo_vectorised_ops()
    demo_matmul()
    csv_statistics(CSV_FILE)
    # 7️⃣ Clean log statement
    print("\n✅ All demonstrations successfully executed.")