import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#######################################
# 1) Matrix-building and time evolution
#######################################

def param_to_matrix(N, x):
    """
    Build an N×N Hermitian Toeplitz matrix M^b from a real parameter array x.
    """
    c = []
    for k in range(N - 1):
        re_k = x[2 * k]
        im_k = x[2 * k + 1]
        c_k = re_k + 1j * im_k
        c.append(c_k)

    M = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i, j] = 0.0
            else:
                d = abs(j - i)
                if j > i:
                    M[i, j] = c[d - 1]
                else:
                    M[i, j] = np.conjugate(c[d - 1])
    return M

def make_dft(N, normalize=True):
    """
    Create an NxN DFT matrix. If normalize=True, divide by sqrt(N) to make it unitary.
    """
    omega = np.exp(-2j * np.pi / N)
    F = np.array([[omega**(i * j) for j in range(N)] for i in range(N)])
    if normalize:
        F /= np.sqrt(N)
    return F

def build_Ub(Mb, gL, t):
    """
    Compute U^b(t) = P * diag(exp(i*gL*w*t)) * P_inv from matrix Mb.
    """
    w, P = np.linalg.eig(Mb)
    exp_diag = np.diag(np.exp(1j * gL * t * w))
    P_inv = np.linalg.inv(P)
    return P @ exp_diag @ P_inv

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
    max_dim = 8
    num_repeats = 20
    gL = 1.0
    t_array = [0.5]

    results = []

    for N in range(2, max_dim + 1):
        print(f"Starting optimization for dimension N = {N}")
        F = make_dft(N, normalize=True)
        best_fidelity = -np.inf
        best_matrix = None

        for run in range(num_repeats):
            print(f"  Run {run + 1}/{num_repeats} for dimension N = {N}")
            x0 = np.random.randn(2 * (N - 1))

            def objective_fun(x):
                Mb = param_to_matrix(N, x)
                cost = 0.0
                for t in t_array:
                    Ub_t = build_Ub(Mb, gL, t)
                    fid = compute_unitary_fidelity(Ub_t, F)
                    cost += (1.0 - fid)
                return cost

            result = minimize(objective_fun, x0, method='BFGS', options={'maxiter': 2000, 'disp': False})
            fidelity = 1.0 - result.fun
            Mb = param_to_matrix(N, result.x)

            results.append({
                "dimension": N,
                "run": run + 1,
                "fidelity": fidelity,
                "arguments": result.x.tolist(),
                "matrix": Mb.tolist()
            })

            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_matrix = Mb

        print(f"Finished optimization for dimension N = {N}")
        print(f"  Best fidelity for N = {best_fidelity:.6f}")
        print(f"  Corresponding matrix for best fidelity:\nReal part:\n{best_matrix.real}\nImaginary part:\n{best_matrix.imag}")

    return results

def save_results_to_excel(results):
    df = pd.DataFrame(results)
    df.to_excel("optimized_results_with_matrices.xlsx", index=False)
    print("Results saved to 'optimized_results_with_matrices.xlsx'.")

#######################################
# 3) Plotting Fidelity vs. Dimension
#######################################

def plot_fidelity_vs_dimension():
    # Load data from Excel
    df = pd.read_excel("optimized_results_with_matrices.xlsx")

    # Compute average fidelity per dimension size
    avg_fidelity = df.groupby("dimension")["fidelity"].mean()

    # Extract x (dimension size) and y (fidelity values)
    x_vals = avg_fidelity.index
    y_vals = avg_fidelity.values

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 3))

    # Plot a single line
    ax.plot(x_vals, y_vals, marker='o', linestyle='-', color='tab:orange', label="Fidelity")

    # Alternating shaded background
    for i in range(len(x_vals)):
        if i % 2 == 0:
            ax.axvspan(x_vals[i] - 0.5, x_vals[i] + 0.5, color='gray', alpha=0.2)

    # Formatting
    ax.set_xlabel("Dimension Size (N)")
    ax.set_ylabel("Fidelity")
    ax.set_title("Fidelity vs. Dimension Size")
    ax.legend()

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
