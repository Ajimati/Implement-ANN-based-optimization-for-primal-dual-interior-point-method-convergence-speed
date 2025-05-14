# Generate the dataset over multiple problems and save to CSV
import pandas as pd
from generating_lp_problem import generate_lp_problem
from simulating_pd_ipm_iterations import simulate_pd_ipm_iterations


def generate_dataset(num_problems=100, filename='pd_ipm_dataset.csv'):
    all_data = []  # Store all data across problems

    for i in range(num_problems):  # Loop over number of LP problems
        A, b, c = generate_lp_problem(n=20, m=10)  # Create a random LP
        data = simulate_pd_ipm_iterations(A, b, c, max_iters=30)  # Simulate and log PD-IPM iterations

        for row in data:  # Add problem ID to each iteration
            row['problem_id'] = i

        all_data.extend(data)  # Append iteration logs to dataset

    df = pd.DataFrame(all_data)  # Convert to DataFrame for easy CSV writing
    df.to_csv(filename, index=False)  # Save dataset to CSV file
    print(f"Dataset saved to {filename} with {len(df)} rows.")  # Notify user detail of the generated dataset
    
# main generation
if __name__ == "__main__":
    generate_dataset(num_problems=400)  # Generate 400 problems' worth of data 
