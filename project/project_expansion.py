import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'prices.csv')
df = pd.read_csv(file_path, header=None)
data = df.values.T # Shape: (1256, 7)

# Normalize
min_val = np.min(data, axis=0)
max_val = np.max(data, axis=0)
data_normalized = (data - min_val) / (max_val - min_val)

# Helper
def create_dataset(dataset, input_cols, target_col, look_back=10, look_ahead=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_ahead + 1):
        a = dataset[i:(i + look_back), input_cols]
        dataX.append(a.flatten())
        dataY.append(dataset[i + look_back + look_ahead - 1, target_col])
    return np.array(dataX), np.array(dataY)

class StockMLP(nn.Module):
    def __init__(self, input_dim):
        super(StockMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.layers(x)

# --- Experiment 1: Hyperparameter Analysis (Look Back Window) ---
print("Running Hyperparameter Analysis...")
look_backs = [5, 10, 20, 30, 50]
mses = []
target_col = 6 # Stock 7
input_cols = [6] # Only use Stock 7 history
train_size = 1000

for lb in look_backs:
    X, y = create_dataset(data_normalized, input_cols, target_col, look_back=lb)
    
    split_idx = train_size - lb
    X_train = torch.tensor(X[:split_idx], dtype=torch.float32)
    y_train = torch.tensor(y[:split_idx], dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X[split_idx:], dtype=torch.float32)
    y_test = y[split_idx:] # Keep as numpy for eval
    
    # Train ensemble
    preds = []
    for _ in range(3): # 3 models for speed
        model = StockMLP(X_train.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for epoch in range(100):
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds.append(model(X_test).numpy().flatten())
            
    avg_pred = np.mean(preds, axis=0)
    
    # Inverse transform
    scale = max_val[target_col] - min_val[target_col]
    min_v = min_val[target_col]
    y_test_inv = y_test * scale + min_v
    avg_pred_inv = avg_pred * scale + min_v
    
    mse = np.mean((avg_pred_inv - y_test_inv)**2)
    mses.append(mse)
    print(f"Look Back: {lb}, MSE: {mse:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(look_backs, mses, marker='o')
plt.title('Effect of Look-back Window on Prediction Error')
plt.xlabel('Look-back Window Size (Days)')
plt.ylabel('MSE')
plt.grid(True)
plt.savefig(os.path.join(script_dir, 'figures/hyperparameter_analysis.png'))
plt.close()

# --- Experiment 2: Baseline Comparison ---
print("Running Baseline Comparison...")
lb = 10
X, y = create_dataset(data_normalized, input_cols, target_col, look_back=lb)
split_idx = train_size - lb
y_test = y[split_idx:]
scale = max_val[target_col] - min_val[target_col]
min_v = min_val[target_col]
y_test_inv = y_test * scale + min_v

# 1. MLP (Best from previous)
# We already have the MSE for lb=10 from the loop above, let's just re-use or re-calc if needed.
# For simplicity, let's assume the value from the loop is `mses[1]` (index 1 is lb=10)
mlp_mse = mses[1]

# 2. Persistence Model: y_hat_{t} = y_{t-1}
# In our dataset X, the last column is the most recent price (t-1)
# X is normalized, so we take the last column, inverse transform it.
# X shape: (n_samples, look_back). Last column is index -1.
X_test_np = X[split_idx:]
persistence_pred_norm = X_test_np[:, -1]
persistence_pred = persistence_pred_norm * scale + min_v
persistence_mse = np.mean((persistence_pred - y_test_inv)**2)

# 3. Moving Average (10 days)
# The input X contains the past 10 days. So we just take the mean of each row.
ma_pred_norm = np.mean(X_test_np, axis=1)
ma_pred = ma_pred_norm * scale + min_v
ma_mse = np.mean((ma_pred - y_test_inv)**2)

print(f"MLP MSE: {mlp_mse:.4f}")
print(f"Persistence MSE: {persistence_mse:.4f}")
print(f"MA(10) MSE: {ma_mse:.4f}")

# Plot Comparison
models = ['MLP', 'Persistence', 'MA(10)']
errors = [mlp_mse, persistence_mse, ma_mse]

plt.figure(figsize=(8, 5))
plt.bar(models, errors, color=['blue', 'green', 'orange'])
plt.title('Model Performance Comparison')
plt.ylabel('MSE')
for i, v in enumerate(errors):
    plt.text(i, v, f'{v:.1f}', ha='center', va='bottom')
plt.savefig(os.path.join(script_dir, 'figures/baseline_comparison.png'))
plt.close()

