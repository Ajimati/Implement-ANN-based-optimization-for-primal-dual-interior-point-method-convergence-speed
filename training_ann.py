import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("pd_ipm_dataset.csv")

# Features and targets
X = df[['primal_norm', 'dual_norm', 'centering_residual', 'complementarity',
        'x_mean', 's_mean', 'y_mean', 'problem_size_n', 'problem_size_m']].values
y_step = df['step_size_guess'].values.reshape(-1, 1)
y_dir = df['direction_guess'].values.reshape(-1, 1)

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

# Split dataset
X_train, X_test, y_step_train, y_step_test, y_dir_train, y_dir_test = train_test_split(
    X_scaled, y_step, y_dir, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_step_train = torch.tensor(y_step_train, dtype=torch.float32)
y_step_test = torch.tensor(y_step_test, dtype=torch.float32)
y_dir_train = torch.tensor(y_dir_train, dtype=torch.float32)
y_dir_test = torch.tensor(y_dir_test, dtype=torch.float32)

# Define ANN model with two outputs
class PDIPMPredictor(nn.Module):
    def __init__(self, input_dim):
        super(PDIPMPredictor, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.step_output = nn.Linear(32, 1)     # Regression head
        self.dir_output = nn.Linear(32, 1)      # Binary classification head

    def forward(self, x):
        shared = self.shared(x)
        step_size = self.step_output(shared)
        direction = torch.sigmoid(self.dir_output(shared))  # Sigmoid for binary output
        return step_size, direction

# Initialize model, loss functions, optimizer
model = PDIPMPredictor(input_dim=X_train.shape[1])
loss_step = nn.MSELoss()
loss_dir = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    pred_step, pred_dir = model(X_train)

    loss1 = loss_step(pred_step, y_step_train)
    loss2 = loss_dir(pred_dir, y_dir_train)
    loss = loss1 + loss2  # Combine losses

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, StepLoss: {loss1.item():.6f}, DirLoss: {loss2.item():.6f}")

# Save the trained model
torch.save(model.state_dict(), "pdipm_predictor.pth")
print(" Model trained and saved successfully as 'pdipm_predictor.pth'")
