import numpy as np
import pandas as pd
def dft_matrix(N: int) -> np.ndarray:
    """
    Create an N×N unitary Discrete Fourier Transform matrix.

    Parameters
    ----------
    N : int
        Size of the DFT matrix.

    Returns
    -------
    np.ndarray
        An N×N complex DFT matrix, normalized by 1/√N.
    """
    ω = np.exp(-2j * np.pi / N)
    matrix = np.array([[ω**(i * j) for j in range(N)] for i in range(N)], dtype=complex)
    return matrix / np.sqrt(N)

def cyclic_pauli_x_matrix(N, shift=1):
    """
    Construct a cyclic Pauli-X (shift) matrix for a given size N and shift value.
    Essentially performs a circular right shift by 'shift' positions.
    """
    return np.roll(np.eye(N), shift, axis=1)

def hadamard_matrix(N):
    """
    Generate a Hadamard-like matrix scaled to size N (only for power-of-2).
    """
    if not (N != 0 and ((N & (N - 1)) == 0)):
        raise ValueError("Hadamard matrix only defined for power-of-2 sizes.")
    H = np.array([[1]])
    while H.shape[0] < N:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(N)
def random_unitary(N, seed=None):
    """
    Generate a random N×N unitary matrix using QR decomposition of a random complex matrix.
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Q, R = np.linalg.qr(Z)
    D = np.diag(np.exp(1j * np.angle(np.diag(R))))
    return Q @ D

def create_T_block(a: int, b: int, right: bool, U: np.ndarray) -> tuple[float, float, np.ndarray]:
    #Build a two-mode beam-splitter (T) block to zero out a specific element in U.

    if right:
        # Zero U[a, b] by right-multiplying T on columns b and b+1
        a1 = U[a, b]
        b1 = U[a, b + 1]
        theta = np.arctan2(abs(a1), abs(b1))
        phi = np.angle(a1) - np.angle(b1)

        # Build the local 2×2 block, then embed into a 5×5 identity
        T_block = np.array([
            [np.exp(1j * phi) * np.cos(theta), -np.sin(theta)],
            [np.exp(1j * phi) * np.sin(theta),  np.cos(theta)]
        ], dtype=complex)
        T = np.eye(5, dtype=complex)
        T[np.ix_([b, b + 1], [b, b + 1])] = T_block.conj().T

    else:
        # Zero U[a, b] by left-multiplying T on rows a-1 and a
        a1 = U[a - 1, b]
        b1 = U[a, b]
        theta = np.arctan2(abs(b1), abs(a1))
        phi = np.angle(b1) - np.angle(a1) - np.pi

        T_block = np.array([
            [np.exp(1j * phi) * np.cos(theta), -np.sin(theta)],
            [np.exp(1j * phi) * np.sin(theta),  np.cos(theta)]
        ], dtype=complex)
        T = np.eye(5, dtype=complex)
        T[np.ix_([a - 1, a], [a - 1, a])] = T_block

    return theta, phi, T


def traverse_diagonals(n: int) -> tuple[list[int], list[int], list[int]]:
    """
    Generate the traversal order of matrix indices along SW–NE diagonals.

    For an n×n matrix, returns three lists:
      - x_list: row indices,
      - y_list: column indices,
      - right_flags: 1 for right-multiply at each step, 0 for left-multiply.

    The sequence starts at (n-1, 0), then zig-zags as in Clements decomposition.
    """
    x = n - 1
    y = 0
    trigger = 0          # 0 = moving up-left, 1 = moving down-right
    count = 0
    x_list, y_list = [], []
    right_flags = []

    # We only need n-1 + n-1 = 2n-2 steps to hit every off-diagonal
    while count <= (n - 2):
        x_list.append(x)
        y_list.append(y)
        right_flags.append(int(trigger == 0))

        if trigger == 0:
            # Climbing up-left
            if y == 0:
                # Hit left border → switch
                trigger = 1
                x -= 1
                count += 1
            else:
                x -= 1
                y -= 1
        else:
            # Descending down-right
            if x == n - 1:
                # Hit bottom border → switch
                trigger = 0
                y += 1
                count += 1
            else:
                x += 1
                y += 1

    return x_list, y_list, right_flags


if __name__ == "__main__":
    # Create the 5×5 unitary DFT matrix to decompose
    N = 5
    U = random_unitary(N)
    data = []
    # Figure out which positions to zero and whether to right- or left-multiply
    row_indices, col_indices, right_flags = traverse_diagonals(N)

    # Apply a sequence of T-blocks to clear off-diagonal elements step by step
    for step, (i, j, right) in enumerate(zip(row_indices, col_indices, right_flags)):
        theta, phi, T = create_T_block(i, j, bool(right), U)

        # Update U: either U @ T (right) or T @ U (left)
        if right:
            U = U @ T
        else:
            U = T @ U

        # Print the current transformation step details
        print(f"Step {step:2d} → cleared element at ({i},{j}), "
              f"{'right' if right else 'left'} multiply")
        print(f"   θ (theta): {theta:.4f} rad")
        print(f"   φ (phi)  : {phi:.4f} rad")
        print("   Updated U:")
        print(np.round(U, 3))
        print()  # blank line for readability




#####################################################
# UnComment if you want saved into a CSV
    #     data.append({
    #         "step": i,
    #         "theta": theta,
    #         "phi": phi,
    #         "T_matrix": T,
    #         "U_matrix": U.copy()
    #     })
    # df = pd.DataFrame(data)
    # csv_path = "unitary_decomposition.csv"
    # df.to_csv(csv_path, index=False)
