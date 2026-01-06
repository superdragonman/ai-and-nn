import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def sin_2pi_on_grid(x):
    return np.sin(2 * np.pi * x)

class MLP(nn.Module):
    def __init__(self, hidden_units=32):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(1, hidden_units)
        self.hidden2 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(hidden_units, 1)
    
    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

def run_extrapolation_experiment():
    print("Running extrapolation experiment...")
    
    # 1. Data Generation (Training Domain: [0, 1])
    num_points = 100
    x_train_np = np.linspace(0, 1, num_points)
    y_train_clean = sin_2pi_on_grid(x_train_np)
    
    # Add noise (Standard scenario)
    noise_std = 0.4
    y_train_noisy = y_train_clean + np.random.normal(0, noise_std, num_points)
    
    # Prepare tensors
    x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32).view(-1, 1)
    y_train_tensor = torch.tensor(y_train_noisy, dtype=torch.float32).view(-1, 1)
    
    # 2. Model Training
    model = MLP(hidden_units=32)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    num_epochs = 2000
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
    # 3. Evaluation (Extrapolation Domain: [-1, 2])
    model.eval()
    x_eval = np.linspace(-1, 2, 400) # Wider range
    x_eval_tensor = torch.tensor(x_eval, dtype=torch.float32).view(-1, 1)
    
    with torch.no_grad():
        y_pred = model(x_eval_tensor).numpy()
        
    # 4. Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot true function over the whole range
    plt.plot(x_eval, sin_2pi_on_grid(x_eval), 'g-', linewidth=2, alpha=0.5, label='True Function (sin(2πx))')
    
    # Plot training data (only in [0, 1])
    plt.scatter(x_train_np, y_train_noisy, color='orange', s=15, alpha=0.6, label='Training Data (x ∈ [0, 1])')
    
    # Plot model prediction
    plt.plot(x_eval, y_pred, 'b--', linewidth=2, label='Model Prediction')
    
    # Highlight training region
    plt.axvspan(0, 1, color='gray', alpha=0.1, label='Training Region')
    
    plt.title('Model Extrapolation Capability Analysis\n(Training on [0, 1], Testing on [-1, 2])')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = 'extrapolation_result.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

if __name__ == "__main__":
    run_extrapolation_experiment()
