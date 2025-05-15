import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Example training data (replace this with your actual training data)
# Shape: (number of samples, number of features)
X_train = np.array([
    [2.35, 1.89, 3.02, 15.6, 1.2, 1.1, 0.8, 20, 10],
    [0.85, 0.74, 1.10, 5.2, 1.05, 1.00, 0.95, 20, 10],
    [0.12, 0.10, 0.08, 0.3, 0.99, 1.01, 0.98, 20, 10],
    # Add more training data here...
])

# Step 1: Fit the scaler on your training data
scaler = StandardScaler()
scaler.fit(X_train)

# Step 2: Save the fitted scaler to a file
joblib.dump(scaler, 'scaler.pkl')

print("Scaler saved as 'scaler.pkl'")
