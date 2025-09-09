import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#######################################
# 1) Matrix-building and time evolution
#######################################

def param_to_matrix(N, x):
    """
    Construct an NxN lower-triangular matrix where:
    
    - Each row (starting from the second) has **two** nonzero elements.
    - The first row has **only one** nonzero element.
    - The pattern follows:
    
        A1   0    0    0   ...
        A2  A3    0    0   ...
        0   A4   A5    0   ...
        0    0   A6   A7   ...
    
    - Requires **(2N - 1) real parameters**.
    - Each complex number is formed as `re + 1j * im` from pairs of real values.

    :param N: Matrix size (NxN)
    :param x: Real parameter array (length = 2N - 1)
    :return: Complex NxN matrix
    """
    required_length = 2 * (2 * N - 1)  # Total required real parameters (ensuring even length)
    if len(x) != required_length:
        raise ValueError(f"Expected x of length {required_length}, got {len(x)}")

    M = np.zeros((N, N), dtype=complex)

    idx = 0  # Track parameter index
    for i in range(N):
        for j in range(i - 1, i + 1):  # Only two consecutive elements per row
            if 0 <= j < N:
                re_val = x[2 * idx]  # Real part
                im_val = x[2 * idx + 1]  # Imaginary part
                M[i, j] = re_val + 1j * im_val
                idx += 1  # Move to next complex number

    return M

def make_cyclic_generalized_pauliX(N, shift=0):
    """
    Generate an NxN cyclic generalized Pauli X matrix with a shift.
    """
    Xc = np.zeros((N, N), dtype=complex)
    for i in range(N):
        Xc[i, (i) % N] = 1
    return np.roll(Xc, shift, axis=1)

def build_Wb_svd(Mb, gL, t):
    """
    Compute time evolution operator W^b(t) using SVD of Mb:
       Mb = U * Sigma * V^\dagger
       W^b(t) = U * diag(exp(i*gL*t*Sigma)) * V^\dagger
    """
    W, svals, Vh = np.linalg.svd(Mb)
    exp_diag = np.diag(np.exp(1j * gL * t * svals))
    Wb_t = W @ exp_diag @ Vh
    return Wb_t

def compute_unitary_fidelity(U, V):
    """
    Compute fidelity = (1/N^2) * |Trace(U^† * V)|^2.
    """
    N = U.shape[0]
    trace_val = np.trace(U.conjugate().T @ V)
    return (1.0 / (N**2)) * np.abs(trace_val)**2

#######################################
# 2) Optimization and Data Storage
#######################################

def optimize_and_save_results():
    max_dim = 10
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
                
                # Generate correct number of random parameters
                x0 = np.random.randn(2 * (2 * N - 1))  # Fixed initialization
                
                def objective_fun(x):
                    Mb = param_to_matrix(N, x)
                    cost = 0.0
                    for t in t_array:
                        Ub_t = build_Wb_svd(Mb, gL, t)
                        fid = compute_unitary_fidelity(Ub_t, Xc_shifted)
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
    df.to_excel("optimized_results_pauliX_One_Frequency_Intermodal.xlsx", index=False)
    print("Results saved to 'optimized_results_pauliX_One_Frequency_Intermodal.xlsx'.")

#######################################
# 3) Plotting Fidelity vs. Dimension (With Shifts)
#######################################

def plot_fidelity_vs_dimension():
    # Load data from Excel
    df = pd.read_excel("optimized_results_pauliX_One_Frequency_Intermodal.xlsx")

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
    ax.set_title("Fidelity vs. Dimension Size IntraModal 1 frequency (Pauli X with Shifts)")
    ax.legend(title="Cyclic Shift")

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
