import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import joblib
import numpy as np

# Load saved scaler
scaler = joblib.load("scaler.pkl")

# Define the ANN model structure
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

# Load the trained model
model = StepSizePredictor()
model.load_state_dict(torch.load("step_size_predictor.pth"))
model.eval()

# Predict function
def predict_step_size(features):
    features_scaled = scaler.transform([features])
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return prediction

# GUI app
class PredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("PD-IPM Step Size Predictor")

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
        self.predict_button = tk.Button(master, text="Predict Step Size", command=self.run_prediction)
        self.predict_button.grid(row=10, columnspan=2, pady=10)

        # Result display
        self.result_label = tk.Label(master, text="Prediction will appear here.")
        self.result_label.grid(row=11, columnspan=2)

    def run_prediction(self):
        try:
            # Get input values from GUI
            inputs = [float(entry.get()) for entry in self.entries]
            # Predict step size
            step_size = predict_step_size(inputs)
            self.result_label.config(text=f"Predicted Step Size: {step_size:.6f}")
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numeric values in all fields.")

# Launch the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()
