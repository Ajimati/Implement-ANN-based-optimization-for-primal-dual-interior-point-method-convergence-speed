import pandas as pd
from generating_lp_problem import generate_lp_problem
from simulating_pd_ipm_iterations import simulate_pd_ipm_iterations

def generate_dataset(num_problems=400, filename='pd_ipm_dataset.csv'):
    all_data = []

    for i in range(num_problems):
        A, b, c = generate_lp_problem(n=20, m=10)
        data = simulate_pd_ipm_iterations(A, b, c, max_iters=30)

        for row in data:
            row['problem_id'] = i
            all_data.append(row)

    df = pd.DataFrame(all_data)
    
    # Ensure all relevant columns are present
    required_columns = [
        'primal_norm', 'dual_norm', 'centering_residual',
        'complementarity', 'x_mean', 's_mean', 'y_mean',
        'problem_size_n', 'problem_size_m',
        'step_size_guess', 'direction_guess'
    ]
    
    # Check and warn if any column is missing
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        print(f"Warning: Missing expected columns: {missing_cols}")
    
    df.to_csv(filename, index=False)
    print(f" Dataset saved to '{filename}' with {len(df)} rows.")

# Run it directly
if __name__ == "__main__":
    generate_dataset(num_problems=400)
