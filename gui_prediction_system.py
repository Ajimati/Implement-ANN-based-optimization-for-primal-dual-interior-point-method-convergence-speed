import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import joblib
import numpy as np

# Load saved scaler
scaler = joblib.load("scaler.pkl")

# Define the ANN model with two outputs (step size + direction)
class PDIPMPredictor(nn.Module):
    def __init__(self):
        super(PDIPMPredictor, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.step_output = nn.Linear(32, 1)     # Step size regression head
        self.dir_output = nn.Linear(32, 1)      # Direction classification head

    def forward(self, x):
        shared = self.shared(x)
        step_size = self.step_output(shared)
        direction = torch.sigmoid(self.dir_output(shared))
        return step_size, direction

# Load trained model
model = PDIPMPredictor()
model.load_state_dict(torch.load("pdipm_predictor.pth"))
model.eval()

# Prediction function for both outputs
def predict(features):
    features_scaled = scaler.transform([features])
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    with torch.no_grad():
        step_pred, dir_pred = model(input_tensor)
    step_size = step_pred.item()
    direction = 1 if dir_pred.item() >= 0.5 else 0  # Threshold at 0.5
    return step_size, direction

# GUI app class
class PredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("PD-IPM Step Size & Direction Predictor")

        # Feature labels and entry boxes
        self.entries = []
        labels = [
            "Primal residual norm",
            "Dual residual norm",
            "Centering residual norm",
            "Complementarity (xᵀs)",
            "Mean of x",
            "Mean of s",
            "Mean of y",
            "Number of variables (n)",
            "Number of constraints (m)"
        ]

        for i, label in enumerate(labels):
            tk.Label(master, text=label).grid(row=i, column=0, sticky='w')
            entry = tk.Entry(master)
            entry.grid(row=i, column=1)
            self.entries.append(entry)

        # Predict button
        self.predict_button = tk.Button(master, text="Predict", command=self.run_prediction)
        self.predict_button.grid(row=10, columnspan=2, pady=10)

        # Result display
        self.result_label = tk.Label(master, text="Prediction will appear here.")
        self.result_label.grid(row=11, columnspan=2)

    def run_prediction(self):
        try:
            inputs = [float(entry.get()) for entry in self.entries]
            step_size, direction = predict(inputs)
            dir_str = "Forward (1)" if direction == 1 else "Backward (0)"
            self.result_label.config(text=f"Predicted Step Size: {step_size:.6f}\nPredicted Direction: {dir_str}")
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numeric values in all fields.")

# Launch the GUI app
if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()
