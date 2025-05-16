import numpy as np

def simulate_pd_ipm_iterations(A, b, c, max_iters=20):
    m, n = A.shape

    x = np.ones(n)
    s = np.ones(n)
    y = np.ones(m)

    mu = 1.0
    tol = 1e-6
    data = []

    previous_complementarity = None

    for k in range(max_iters):
        r_dual = A.T @ y + s - c
        r_primal = A @ x - b
        r_cent = np.multiply(x, s) - mu * np.ones(n)

        residual_norm = np.linalg.norm(np.concatenate([r_dual, r_primal, r_cent]))
        if residual_norm < tol:
            break

        complementarity = np.dot(x, s)

        # Determine direction 
        if previous_complementarity is not None:
            direction_guess = 1 if complementarity < previous_complementarity else 0
        else:
            direction_guess = 1  # First iteration assumed positive

        data.append({
            'iteration': k,
            'primal_norm': np.linalg.norm(r_primal),
            'dual_norm': np.linalg.norm(r_dual),
            'centering_residual': np.linalg.norm(r_cent),
            'complementarity': complementarity,
            'step_size_guess': mu,
            'x_mean': np.mean(x),
            's_mean': np.mean(s),
            'y_mean': np.mean(y),
            'problem_size_n': n,
            'problem_size_m': m,
            'direction_guess': direction_guess
        })

        previous_complementarity = complementarity

        # Simulate basic updates
        x = x - 0.01 * r_dual[:n]
        y = y - 0.01 * r_primal
        s = s - 0.01 * r_dual[:n]

        mu *= 0.9

    return data
