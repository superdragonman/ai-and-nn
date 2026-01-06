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
data = df.values.T # Shape (1256, 7)

# Normalize
min_val = np.min(data, axis=0)
max_val = np.max(data, axis=0)
data_normalized = (data - min_val) / (max_val - min_val)

# Helper to create dataset
def create_dataset(dataset, input_cols, target_col, look_back=10, look_ahead=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_ahead + 1):
        a = dataset[i:(i + look_back), input_cols]
        dataX.append(a.flatten())
        dataY.append(dataset[i + look_back + look_ahead - 1, target_col])
    return np.array(dataX), np.array(dataY)

# Define MLP Model
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

def train_model(X_train, y_train, input_dim, epochs=100):
    model = StockMLP(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    return model

# --- Experiment 1: Hyperparameter Analysis (Look Back Window) ---
print("Running Hyperparameter Analysis...")
look_backs = [5, 10, 20, 30, 50]
mses = []
target_col = 6 # Stock 7
input_cols = [6] # Only use Stock 7 history (since it was the best)

train_size = 1000

for lb in look_backs:
    X, y = create_dataset(data_normalized, input_cols, target_col, look_back=lb)
    
    # Split
    # Note: The dataset size changes with look_back, so we need to be careful with indices
    # But for simplicity, let's just take the last part as test
    # Actually, to be fair, we should keep the test set fixed in time.
    # Total length is 1256. Let's say test is last 256 points.
    
    test_len = 256
    train_len = len(X) - test_len
    
    X_train, X_test = X[:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]
    
    # Train ensemble of 5 models
    ensemble_preds = []
    for _ in range(5):
        model = train_model(X_train, y_train, input_dim=lb, epochs=100)
        model.eval()
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_test)).numpy().flatten()
            # Inverse transform
            pred_inv = pred * (max_val[target_col] - min_val[target_col]) + min_val[target_col]
            ensemble_preds.append(pred_inv)
            
    avg_pred = np.mean(ensemble_preds, axis=0)
    y_test_inv = y_test * (max_val[target_col] - min_val[target_col]) + min_val[target_col]
    
    mse = np.mean((avg_pred - y_test_inv)**2)
    mses.append(mse)
    print(f"Look Back: {lb}, MSE: {mse:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(look_backs, mses, marker='o')
plt.title('Effect of Look Back Window on Prediction Error')
plt.xlabel('Look Back Window Size (Days)')
plt.ylabel('MSE')
plt.grid(True)
plt.savefig(os.path.join(script_dir, 'figures/hyperparameter_analysis.png'))
plt.close()

# --- Experiment 2: Baseline Comparison ---
print("Running Baseline Comparison...")
# Persistence Model: Predict tomorrow = today
# For the test set used in the best model (look_back=10)
lb = 10
X, y = create_dataset(data_normalized, input_cols, target_col, look_back=lb)
test_len = 256
train_len = len(X) - test_len
y_test = y[train_len:]
y_test_inv = y_test * (max_val[target_col] - min_val[target_col]) + min_val[target_col]

# The "today" value is the last value in the input window.
# X has shape (samples, look_back). The last column is x_{t-1}.
X_test = X[train_len:]
last_val_normalized = X_test[:, -1] 
last_val_inv = last_val_normalized * (max_val[target_col] - min_val[target_col]) + min_val[target_col]

mse_baseline = np.mean((last_val_inv - y_test_inv)**2)
print(f"Baseline MSE: {mse_baseline:.4f}")

# Compare with MLP (look_back=10)
mse_mlp = mses[1] # Index 1 corresponds to look_back=10
print(f"MLP MSE: {mse_mlp:.4f}")

# Bar chart
plt.figure(figsize=(6, 5))
models = ['Persistence (Baseline)', 'MLP (Ours)']
errors = [mse_baseline, mse_mlp]
plt.bar(models, errors, color=['gray', 'blue'])
plt.title('Model Comparison')
plt.ylabel('MSE')
for i, v in enumerate(errors):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
plt.savefig(os.path.join(script_dir, 'figures/baseline_comparison.png'))
plt.close()
