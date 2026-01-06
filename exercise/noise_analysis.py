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

def run_experiment(input_noise_std, output_noise_std, scenario_name):
    print(f"Running experiment: {scenario_name}")
    
    # 1. Data Generation
    num_points = 100
    x_true = np.linspace(0, 1, num_points)
    y_true = sin_2pi_on_grid(x_true)
    
    # Add noise
    x_train = x_true + np.random.normal(0, input_noise_std, num_points)
    y_train = y_true + np.random.normal(0, output_noise_std, num_points)
    
    # Prepare tensors
    x_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    # 2. Model Training
    model = MLP(hidden_units=32)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    num_epochs = 2000 # Increased epochs to ensure convergence/overfitting
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
    # 3. Evaluation
    model.eval()
    # Evaluate on a clean, dense grid to see the learned function
    x_eval = np.linspace(0, 1, 200)
    x_eval_tensor = torch.tensor(x_eval, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        y_pred = model(x_eval_tensor).numpy()
        
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(x_train, y_train, color='orange', s=10, alpha=0.6, label='Training Data')
    
    # Plot true function
    plt.plot(x_eval, sin_2pi_on_grid(x_eval), 'g-', linewidth=2, label='True Function')
    
    # Plot model prediction
    plt.plot(x_eval, y_pred, 'b--', linewidth=2, label='Model Prediction')
    
    plt.title(f'Scenario: {scenario_name}\n(Input Noise $\sigma$={input_noise_std}, Output Noise $\sigma$={output_noise_std})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f'exp_{scenario_name}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

# Define scenarios
scenarios = [
    (0.0, 0.4, "baseline_output_noise"),   # Original case
    (0.1, 0.0, "input_noise_only"),        # Input noise only
    (0.1, 0.4, "both_noise"),              # Both
    (0.0, 0.8, "high_output_noise")        # High output noise
]

if __name__ == "__main__":
    for in_std, out_std, name in scenarios:
        run_experiment(in_std, out_std, name)
