import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load data
# The file has no header, so we let pandas assign default integer columns
# But actually the file format is rows of stocks.
# We want columns to be stocks and rows to be time steps for easier processing.
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'prices.csv')
df = pd.read_csv(file_path, header=None)
data = df.values.T # Shape becomes (1256, 7)
print(f"Data shape: {data.shape}")

# Normalize data (MinMax Scaling)
# It's good practice for MLPs
min_val = np.min(data, axis=0)
max_val = np.max(data, axis=0)
data_normalized = (data - min_val) / (max_val - min_val)

# --- EDA: Plot All Stocks ---
plt.figure(figsize=(12, 6))
for i in range(7):
    plt.plot(data[:, i], label=f'Stock {i+1}')
plt.title('Price History of All 7 Stocks')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(script_dir, 'figures/all_stocks_history.png'))
plt.close()

# --- EDA: Correlation Matrix ---
plt.figure(figsize=(10, 8))
corr_matrix = np.corrcoef(data.T)

# Use a sequential colormap (YlOrRd) which is better for positive correlations
# Auto-scale the range to the data (or slightly wider) to maximize contrast
# Since correlations are high (0.68-1.0), we set vmin=0.6 to make the lowest values distinct from 0
plt.imshow(corr_matrix, cmap='YlOrRd', interpolation='nearest', vmin=0.6, vmax=1.0)
plt.colorbar()
plt.title('Stock Correlation Matrix')
tick_marks = np.arange(7)
plt.xticks(tick_marks, [f'S{i+1}' for i in range(7)])
plt.yticks(tick_marks, [f'S{i+1}' for i in range(7)])

# Add text annotations
for i in range(7):
    for j in range(7):
        val = corr_matrix[i, j]
        # For YlOrRd, high values (Red) need white text, low values (Yellow) need black text
        # Threshold around 0.85 seems appropriate for YlOrRd
        color = 'white' if val > 0.85 else 'black'
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'figures/stock_correlation.png'))
plt.close()

# Parameters
look_back = 10
look_ahead = 1
train_size = 1000
test_size = len(data) - train_size

# Helper to create dataset
def create_dataset(dataset, input_cols, target_col, look_back=10, look_ahead=1):
    dataX, dataY = [], []
    # dataset shape: (num_samples, num_features)
    # input_cols: list of column indices to use as input
    # target_col: column index to predict
    
    for i in range(len(dataset) - look_back - look_ahead + 1):
        # Input: dataset[i : i+look_back, input_cols]
        # We flatten this to a 1D vector
        a = dataset[i:(i + look_back), input_cols]
        dataX.append(a.flatten())
        
        # Output: dataset[i + look_back + look_ahead - 1, target_col]
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

def train_and_evaluate(task_name, input_cols, target_col):
    print(f"\n--- Processing {task_name} ---")
    
    # Create dataset
    X, y = create_dataset(data_normalized, input_cols, target_col, look_back, look_ahead)
    
    # Split into train and test
    split_idx = train_size - look_back
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    print(f"Train X shape: {X_train.shape}, Y shape: {y_train.shape}")
    print(f"Test X shape: {X_test.shape}, Y shape: {y_test.shape}")
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Ensemble Training
    n_models = 5
    predictions = []
    
    print(f"Training ensemble of {n_models} models...")
    for i in range(n_models):
        # Initialize model with random weights
        input_dim = X_train.shape[1]
        model = StockMLP(input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train
        epochs = 200
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(X_test_t).numpy().flatten()
            predictions.append(pred)
            
    # Calculate Ensemble Statistics
    predictions = np.array(predictions) # Shape: (n_models, n_test_samples)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # Inverse transform
    scale = max_val[target_col] - min_val[target_col]
    min_v = min_val[target_col]
    
    y_test_actual = y_test * scale + min_v
    mean_pred_actual = mean_pred * scale + min_v
    std_pred_actual = std_pred * scale # Scale std deviation too
    
    # Calculate MSE on mean prediction
    mse_actual = np.mean((y_test_actual - mean_pred_actual)**2)
    print(f"Ensemble Test MSE (Actual): {mse_actual:.4f}")
    
    # Plot with Uncertainty
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='True Value', color='black', alpha=0.7)
    plt.plot(mean_pred_actual, label='Mean Prediction', color='blue', linestyle='--', alpha=0.8)
    
    # 95% Confidence Interval (Mean +/- 1.96 * Std)
    lower_bound = mean_pred_actual - 1.96 * std_pred_actual
    upper_bound = mean_pred_actual + 1.96 * std_pred_actual
    plt.fill_between(range(len(mean_pred_actual)), lower_bound, upper_bound, color='blue', alpha=0.2, label='95% Confidence Interval')
    
    plt.legend()
    plt.title(f'Stock 7 Prediction with Uncertainty - {task_name}')
    plt.xlabel('Time Step (Test Set)')
    plt.ylabel('Price')
    
    # Save plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        
    filename = f'stock_prediction_{task_name.split()[1]}.png'
    plt.savefig(os.path.join(figures_dir, filename))
    print(f"Plot saved to figures/{filename}")
    plt.close()

# Task 1: Input = Stocks 0-5, Output = Stock 6
train_and_evaluate("Task 1", list(range(6)), 6)

# Task 2: Input = Stock 6, Output = Stock 6
train_and_evaluate("Task 2", [6], 6)

# Task 3: Input = Stocks 0-6, Output = Stock 6
train_and_evaluate("Task 3", list(range(7)), 6)
