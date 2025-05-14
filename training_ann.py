# Ann is trained on the generated pd-ipm dataset
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("pd_ipm_dataset.csv")

# Features and target (we predict step_size_guess)
X = df[['primal_norm', 'dual_norm', 'centering_residual', 'complementarity', 'x_mean', 's_mean', 'y_mean', 'problem_size_n', 'problem_size_m']].values
y = df['step_size_guess'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the ANN model
class StepSizePredictor(nn.Module):
    def __init__(self):
        super(StepSizePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(X_train.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# Initialize model, loss, and optimizer
model = StepSizePredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Save the  developed model
torch.save(model.state_dict(), "step_size_predictor.pth")
print("Model saved successfully!")
