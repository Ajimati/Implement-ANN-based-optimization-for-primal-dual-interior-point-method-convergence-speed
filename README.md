# Step Size Predictor for Primal-Dual Interior Point Method (PD-IPM)
## Overview
This project implements an Artificial Neural Network (ANN) to predict the step size in the Primal-Dual Interior Point Method (PD-IPM) optimization algorithm. The goal is to optimize the convergence speed of the PD-IPM algorithm by predicting the optimal step size at each iteration based on various feature inputs.

### The system includes:

- A command-line interface (CLI) that allows you to input feature values for prediction.

- A Graphical User Interface (GUI) using Tkinter, which allows users to input features and get predictions visually.

### Features
- Predicts the step size based on 9 input features.

- Trained using historical data from PD-IPM runs.

- Built using PyTorch for the neural network, and scikit-learn for data scaling.

### Prerequisites
Before running the project, make sure you have the following Python libraries installed:

- numpy
- pandas
- torch
- scikit-learn
- joblib
- tkinter 

#### You can install all the required dependencies by running:
pip install -r requirements.txt

### Setup
- Clone this repository or download the source code.
- Ensure you have the model file (step_size_predictor.pth) and the scaler file (scaler.pkl) from the training process.
- If these files are missing, you can train the model by running the training script and saving both the trained model and scaler.

### Training the Model
If you'd like to train the model from scratch, follow these steps:

- Prepare your training data (ensure the data includes 9 feature columns corresponding to PD-IPM parameters).
- Run the training script to train the model and save both the model and scaler files.
python train_model.py

#### This will create:
- step_size_predictor.pth (trained model)
- scaler.pkl (scaler for feature scaling)

### Running the Prediction System
 python predict_step_size.py

#### You'll be prompted to enter the following features from a PD-IPM iteration:

- Primal residual norm
- Dual residual norm
- Centering residual norm
- Complementarity (xᵀ * s)
- Mean of x (primal variables)
- Mean of s (slack variables)
- Mean of y (dual variables)
- Number of variables (n)
- Number of constraints (m)

The model will then predict the step size.

### Graphical User Interface (GUI)
To run the GUI version:

python gui_predictor.py

A window will open where you can input the required features and click "Predict" to get the step size prediction.

### Files in the Project
- train_model.py: Script for training the ANN model and saving the trained model and scaler.
- predict_step_size.py: CLI script for step size prediction.
- gui_predictor.py: GUI script for step size prediction.
- step_size_predictor.pth: Trained ANN model file.
- scaler.pkl: Scaler file for feature scaling.

### How the Model Works
- Feature Input: The model takes 9 features that describe the current state of the PD-IPM algorithm.
- Model Architecture: A feedforward neural network (ANN) with:
- Input layer: 9 features
- Hidden layers: 2 hidden layers (64 and 32 neurons)
- Output layer: Single output representing the predicted step size

Training: The model is trained using historical data of PD-IPM iterations. The training data includes the 9 features and the step size for each iteration.

#### Launch the GUI with python gui_predictor.py.

Input values for the following:

- Primal residual norm
- Dual residual norm
- Centering residual norm
- Complementarity (xᵀ * s)
- Mean of x (primal variables)
- Mean of s (slack variables)
- Mean of y (dual variables)
- Number of variables (n)
- Number of constraints (m)

Click "Predict" to see the predicted step size.

### Contributing
Feel free to fork this project and contribute by submitting issues, pull requests, or feedback. Contributions are always welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.
