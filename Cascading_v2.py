import numpy as np

def dft_matrix(N):
    ω = np.exp(-2j * np.pi / N)
    return np.array([[ω**(i*j) for j in range(N)] for i in range(N)], dtype=complex)/np.sqrt(N)

def two_by_two_unitary(theta, phi):
    return np.array([
        [np.exp(1j*phi)*np.cos(theta), -np.sin(theta)],
        [np.exp(1j*phi)*np.sin(theta),  np.cos(theta)]
    ], dtype=complex)

def embed_unitary(N, i, j, block):
    T = np.eye(N, dtype=complex)
    T[np.ix_([i,j],[i,j])] = block
    return T

def givens_rotation(a, b):
    if abs(b) < 1e-16:
        return 0.0, 0.0
    θ = np.arctan2(abs(b), abs(a))
    φ = np.angle(a) - np.angle(b)
    return θ, φ

def zero_element_left(U, tgt_row, tgt_col):
    """
    Left multiply by T_{i,j} to zero U[tgt_row, tgt_col], mixing rows i=tgt_row-1, j=tgt_row.
    Prints all diagnostics and returns (U_new, T, (i,j,θ,φ)).
    """
    i, j = tgt_row-1, tgt_row
    print(f"\n--- LEFT zero U[{tgt_row},{tgt_col}] by mixing rows {i} & {j} ---")
    print("U before:\n", np.round(U,3))
    a, b = U[i, tgt_col], U[j, tgt_col]
    print(f"Targeting a=U[{i},{tgt_col}] = {a:.4f}, b=U[{j},{tgt_col}] = {b:.4f}")
    θ, φ = givens_rotation(a, b)
    print(f"Computed θ={θ:.4f}, φ={φ:.4f}")
    block = two_by_two_unitary(θ, φ)
    print("2×2 block:\n", np.round(block,3))
    T = embed_unitary(U.shape[0], i, j, block)
    print("Full T:\n", np.round(T,3))
    U_new = T @ U
    print("T @ U =\n", np.round(U_new,3))
    z = U_new[j, tgt_col]
    print(f"Result U[{j},{tgt_col}] = {z:.4e} {'OK' if abs(z)<1e-6 else '⚠️'}")
    return U_new, T, (i, j, θ, φ)

def zero_element_right(U, tgt_row, tgt_col):
    """
    Right multiply by T^{-1}_{k,ℓ} to zero U[tgt_row, tgt_col], mixing cols k=tgt_col, ℓ=tgt_col+1.
    Prints diagnostics and returns (U_new, T_inv, (k,ℓ,θ,φ)).
    """
    print(f"\n--- RIGHT zero U[{tgt_row},{tgt_col}] by mixing cols {tgt_col} & {tgt_col+1} ---")
    print("U before:\n", np.round(U,3))
    # Work on transpose
    U_T = U.T
    # zero in transpose at (tgt_col, tgt_row) via left
    U_Tn, T_left, (i, j, θ, φ) = zero_element_left(U_T, tgt_row=tgt_col, tgt_col=tgt_row)
    T_inv = T_left.T.conj()
    print("Derived T⁻¹ =\n", np.round(T_inv,3))
    # apply to original
    U_new = U @ T_inv
    print("U @ T⁻¹ =\n", np.round(U_new,3))
    z = U_new[tgt_row, tgt_col]
    print(f"Result U[{tgt_row},{tgt_col}] = {z:.4e} {'OK' if abs(z)<1e-6 else '⚠️'}")
    return U_new, T_inv, (i, j, θ, φ)

def decompose(U):
    N = U.shape[0]
    U_work = U.copy()
    print("Initial U:\n", np.round(U_work,3))
    for i in range(1, N):
        if i % 2 == 1:
            for j in range(i):
                row = N-1-j
                col = i-1-j
                U_work, Tinv, params = zero_element_right(U_work, row, col)
        else:
            for j in range(1, i+1):
                row = N + j - i - 1
                col = j
                U_work, T, params = zero_element_left(U_work, row, col)
    print("\nFinal U:\n", np.round(U_work,3))
    return U_work

if __name__ == "__main__":
    U0 = dft_matrix(5)
    decompose(U0)
