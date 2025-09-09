import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#######################################
# 1) Matrix-building and time evolution
#######################################

def param_to_matrix(N, x):
    """
    Build an N×N COMPLEX matrix M^b from a real parameter array x,
    with length 2*N*N. No special symmetry or Toeplitz constraints.
    """
    if len(x) != 2 * N * N:
        raise ValueError(f"Expected x of length {2 * N * N}, got {len(x)}")
    M = np.zeros((N, N), dtype=complex)
    idx = 0
    for i in range(N):
        for j in range(N):
            re_val = x[2 * idx]
            im_val = x[2 * idx + 1]
            M[i, j] = re_val + 1j * im_val
            idx += 1
    return M

def make_cyclic_generalized_pauliX(N, shift=0):
    """
    Generate the N×N cyclic generalized Pauli X matrix with a shift.
    """
    Xc = np.zeros((N, N), dtype=complex)
    for i in range(N):
        Xc[i, (i ) % N] = 1  # Base cyclic shift
    return np.roll(Xc, shift, axis=1)  # Apply additional shift

def build_Wb_svd(Mb, gL, t):
    """
    Given a matrix Mb (N×N), define U^b(t) via its SVD:
       Mb = U * Sigma * V^\dagger
    Then:
       W^b(t) = U * diag(exp(i*gL*t*Sigma)) * V^\dagger,
    where exp(...) acts on the diagonal singular values.

    Returns an NxN complex matrix W^b(t).
    """
    W, svals, Vh = np.linalg.svd(Mb)  # Mb = W Sigma V^\dagger
    exp_diag = np.diag(np.exp(1j * gL * t * svals))
    Wb_t = W @ exp_diag @ Vh
    return Wb_t

def compute_unitary_fidelity(W, V):
    """
    Fidelity(W, V) = (1/N^2) * |trace(W^dagger V)|^2.
    """
    N = W.shape[0]
    trace_val = np.trace(W.conjugate().T @ V)
    return (1.0 / (N**2)) * np.abs(trace_val)**2

#######################################
# 2) Optimization and Data Storage
#######################################

def optimize_and_save_results():
    max_dim = 11
    num_repeats = 10
    gL = 1.0
    t_array = [0.5]

    results = []

    for N in range(2, max_dim + 1):
        print(f"\n=== Starting optimization for dimension N = {N} ===")
        
        for shift in range(N):  # Iterate over all cyclic shifts
            Xc_shifted = make_cyclic_generalized_pauliX(N, shift)
            print(f"\nTarget Cyclic Generalized Pauli X Matrix (Shift={shift}) for N={N}:\n{Xc_shifted}")

            best_fidelity = -np.inf
            best_matrix = None

            for run in range(num_repeats):
                print(f"  Run {run + 1}/{num_repeats} for dimension N = {N}, shift = {shift}")
                x0 = np.random.randn(2 * N * N)  # Fully general NxN complex matrix

                def objective_fun(x):
                    Mb = param_to_matrix(N, x)
                    cost = 0.0
                    for t in t_array:
                        Wb_t = build_Wb_svd(Mb, gL, t)
                        fid = compute_unitary_fidelity(Wb_t, Xc_shifted)
                        cost += (1.0 - fid)
                    return cost

                result = minimize(objective_fun, x0, method='BFGS', options={'maxiter': 2000, 'disp': False})
                fidelity = 1.0 - result.fun
                Mb = param_to_matrix(N, result.x)

                results.append({
                    "dimension": N,
                    "shift": shift,
                    "run": run + 1,
                    "fidelity": fidelity,
                    "arguments": result.x.tolist(),
                    "matrix": Mb.tolist()
                })

                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    best_matrix = Mb

            print(f"\nBest fidelity for N = {N}, shift = {shift}: {best_fidelity:.6f}")
            print(f"  Best matrix for shift={shift}:\nReal part:\n{best_matrix.real}\nImaginary part:\n{best_matrix.imag}")

    return results

def save_results_to_excel(results):
    df = pd.DataFrame(results)
    df.to_excel("optimized_results_pauliX_all_to_all.xlsx", index=False)
    print("Results saved to 'optimized_results_pauliX_all_to_all.xlsx'.")

#######################################
# 3) Plotting Fidelity vs. Dimension (With Shifts)
#######################################

def plot_fidelity_vs_dimension():
    # Load data from Excel
    df = pd.read_excel("optimized_results_pauliX_all_to_all.xlsx")

    # Compute average fidelity per shift and dimension
    grouped = df.groupby(["dimension", "shift"])["fidelity"].mean().reset_index()

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 3))

    # Unique shifts to plot separately
    shifts = grouped["shift"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(shifts)))

    for i, shift in enumerate(shifts):
        subset = grouped[grouped["shift"] == shift]
        ax.plot(subset["dimension"], subset["fidelity"], marker='o', linestyle='-', color=colors[i], label=f"Shift={shift}")

    # Alternating shaded background
    x_vals = grouped["dimension"].unique()
    for i in range(len(x_vals)):
        if i % 2 == 0:
            ax.axvspan(x_vals[i] - 0.5, x_vals[i] + 0.5, color='gray', alpha=0.2)

    # Formatting
    ax.set_xlabel("Dimension Size (N)")
    ax.set_ylabel("Fidelity")
    ax.set_title("Fidelity vs. Dimension Size (DFT Comparison)")
    ax.legend(title="Cyclic Shift")
    ax.set_ylim(0, 1)
    # Show the plot
    plt.show()

#######################################
# 4) Main Execution
#######################################

def main():
    print("Starting the optimization process...")
    results = optimize_and_save_results()
    save_results_to_excel(results)
    plot_fidelity_vs_dimension()

if __name__ == "__main__":
    main()
