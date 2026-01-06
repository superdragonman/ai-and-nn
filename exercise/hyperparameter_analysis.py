import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def get_data(num_points=100, noise_std=0.4):
    x = np.linspace(0, 1, num_points)
    y_true = np.sin(2 * np.pi * x)
    y_noisy = y_true + np.random.normal(0, noise_std, num_points)
    return x, y_true, y_noisy

class FlexibleMLP(nn.Module):
    def __init__(self, hidden_units=32, depth=2, activation='tanh'):
        super(FlexibleMLP, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(1, hidden_units))
        layers.append(self._get_activation(activation))
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(self._get_activation(activation))
            
        # Output layer
        layers.append(nn.Linear(hidden_units, 1))
        
        self.net = nn.Sequential(*layers)
    
    def _get_activation(self, name):
        if name.lower() == 'relu':
            return nn.ReLU()
        elif name.lower() == 'tanh':
            return nn.Tanh()
        elif name.lower() == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, x):
        return self.net(x)

def train_and_evaluate(model, x_train, y_train, num_epochs=2000):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    x_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
    model.eval()
    x_eval = np.linspace(0, 1, 200)
    x_eval_tensor = torch.tensor(x_eval, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        y_pred = model(x_eval_tensor).numpy()
        
    return loss_history, x_eval, y_pred

def plot_comparison(results, title, filename, x_train=None, y_train=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Predictions
    if x_train is not None and y_train is not None:
        ax1.scatter(x_train, y_train, color='gray', alpha=0.3, label='Noisy Data')
    
    x_true = np.linspace(0, 1, 200)
    ax1.plot(x_true, np.sin(2*np.pi*x_true), 'k--', alpha=0.5, label='True Function')
    
    for label, res in results.items():
        loss_hist, x_eval, y_pred = res
        ax1.plot(x_eval, y_pred, label=label, linewidth=2)
        ax2.plot(loss_hist, label=label)
        
    ax1.set_title(f'{title} - Predictions')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title(f'{title} - Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def run_experiments():
    # Base config
    base_points = 100
    base_hidden = 32
    base_depth = 2
    base_act = 'tanh' 
    
    # 1. Hidden Units
    print("Exp 1: Hidden Units")
    x, y_true, y_noisy = get_data(num_points=base_points)
    results = {}
    for h in [4, 32, 128]:
        model = FlexibleMLP(hidden_units=h, depth=base_depth, activation=base_act)
        results[f'Hidden={h}'] = train_and_evaluate(model, x, y_noisy)
    plot_comparison(results, 'Effect of Hidden Units', 'exp_hidden_units.png', x, y_noisy)
    
    # 2. Data Density
    print("Exp 2: Data Density")
    results = {}
    densities = [20, 100, 500]
    for d in densities:
        x_d, _, y_d = get_data(num_points=d)
        model = FlexibleMLP(hidden_units=base_hidden, depth=base_depth, activation=base_act)
        results[f'N={d}'] = train_and_evaluate(model, x_d, y_d)
    plot_comparison(results, 'Effect of Data Density', 'exp_data_density.png') 
    
    # 3. Activation Function
    print("Exp 3: Activation Function")
    x, y_true, y_noisy = get_data(num_points=base_points)
    results = {}
    for act in ['relu', 'tanh', 'sigmoid']:
        model = FlexibleMLP(hidden_units=base_hidden, depth=base_depth, activation=act)
        results[f'Act={act}'] = train_and_evaluate(model, x, y_noisy)
    plot_comparison(results, 'Effect of Activation Function', 'exp_activation.png', x, y_noisy)
    
    # 4. Network Depth
    print("Exp 4: Network Depth")
    x, y_true, y_noisy = get_data(num_points=base_points)
    results = {}
    for d in [1, 3, 6]:
        model = FlexibleMLP(hidden_units=base_hidden, depth=d, activation=base_act)
        results[f'Depth={d}'] = train_and_evaluate(model, x, y_noisy)
    plot_comparison(results, 'Effect of Network Depth', 'exp_depth.png', x, y_noisy)

if __name__ == "__main__":
    run_experiments()
