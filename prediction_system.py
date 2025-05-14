import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Define the ANN model class 
class StepSizePredictor(nn.Module):
    def __init__(self):
        super(StepSizePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# Prompt the user to input required features
def get_user_input():
    print("\nEnter the following features from a PD-IPM iteration:")
    features = []
    prompts = [
        "1. Primal residual norm",
        "2. Dual residual norm",
        "3. Centering residual norm",
        "4. Complementarity (x^T * s)",
        "5. Mean of x (primal variables)",
        "6. Mean of s (slack variables)",
        "7. Mean of y (dual variables)",
        "8. Number of variables (n)",
        "9. Number of constraints (m)"
    ]

    for prompt in prompts:
        while True:
            try:
                val = float(input(f"{prompt}: "))
                features.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    return features

# Prediction function
def predict_step_size(input_features):
    # Load model
    model = StepSizePredictor()
    model.load_state_dict(torch.load("step_size_predictor.pth"))
    model.eval()

    # Load or fit a new scaler 
    scaler = StandardScaler()
    scaler.fit([input_features])  
    input_scaled = scaler.transform([input_features])

    # Convert to tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor).item()

    return prediction

# Main entry point
if __name__ == "__main__":
    user_features = get_user_input()
    predicted_step = predict_step_size(user_features)
    print(f"\n Predicted Step Size: {predicted_step:.6f}")
