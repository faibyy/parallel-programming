import numpy as np
from pathlib import Path

script_location = Path(__file__).resolve().parent
SIZES = [200, 400, 800, 1200, 1600,2000]

def read_matrix(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        n = int(f.readline().strip())
        data = [list(map(float, f.readline().split())) for _ in range(n)]
    return np.array(data, dtype=np.float64)

def verify_result(n: int) -> bool:

    a_path = script_location / f"A_{n}.txt"
    b_path = script_location / f"B_{n}.txt"
    c_path = script_location / f"result_{n}.txt"

    for p in (a_path, b_path, c_path):
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    a_matrix = read_matrix(a_path)
    b_matrix = read_matrix(b_path)
    c_matrix = read_matrix(c_path)

    if a_matrix.shape != (n, n) or b_matrix.shape != (n, n) or c_matrix.shape != (n, n):
        raise ValueError(
            f"Incorrect matrix sizes"
            f"A={a_matrix.shape}, B={b_matrix.shape}, C={c_matrix.shape}, expected {(n, n)}"
        )

    c_reference = a_matrix @ b_matrix
    correct = np.allclose(c_matrix, c_reference)

    if correct:
        return True
    return False

def main():
    all_correct = True
    for n in SIZES:
        try:
            correct = verify_result(n)
            if correct:
                print(f"{n}x{n}: Correct")
            else:
                all_correct = False
                print(f"{n}x{n}: Incorrect")
        except Exception as e:
            all_correct = False
            print(f"{n}x{n}: Error | {e}")

    print("All correct" if all_correct else "Found errors")

if __name__ == "__main__":
    main()