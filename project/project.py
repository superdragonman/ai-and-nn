from numpy import linspace, concatenate, array
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Ensure figures directory exists
script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

#%%
# 1. Generate Lorenz System Data
def lorenz(t, xyz):
    x, y, z = xyz
    s, r, b = 10, 28, 8/3. # parameters Lorentz used
    return [s*(y-x), x*(r-z) - y, x*y - b*z]

a, b = 0, 40
t = linspace(a, b, 4000)

sol = solve_ivp(lorenz, [a, b], [1,1,1], t_eval=t)
data = sol.y.T
train = data[:3000]
test = data[3000:]
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Plot 3D Lorenz Attractor
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5)
ax.set_title("Lorenz Attractor (Generated Data)")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.savefig(os.path.join(figures_dir, 'lorenz_3d.png'))
plt.close()

# %%
# 2. Data preparation for the model
def create_dataset(dataset, look_back=7):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        # Window of past 7 points: i to i+look_back-1
        # Target: i+look_back
        window = dataset[i : i+look_back]
        
        # We want x_{n-1}, ..., x_{n-7} and y_{n-1}, ..., y_{n-7}
        # window contains [x, y, z] at t_{n-7}, ..., t_{n-1} (in chronological order)
        # So window[-1] is t_{n-1}, window[0] is t_{n-7}
        # We reverse them to get n-1 first
        x_vals = window[:, 0][::-1]
        y_vals = window[:, 1][::-1]
        
        # Concatenate to form 14-dim vector
        dataX.append(concatenate((x_vals, y_vals)))
        dataY.append(dataset[i + look_back, 2]) # z is index 2
        
    return array(dataX), array(dataY)

look_back = 7
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print(f"Train X shape: {trainX.shape}")
print(f"Train Y shape: {trainY.shape}")

# Convert to PyTorch tensors
# Note: trainY needs to be (N, 1) for MSELoss with (N, 1) output
X_train_tensor = torch.tensor(trainX, dtype=torch.float32)
y_train_tensor = torch.tensor(trainY, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(testX, dtype=torch.float32)
y_test_tensor = torch.tensor(testY, dtype=torch.float32).unsqueeze(1)

# %%
# 3. Define and Train MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP(input_dim=14)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 200
train_losses = []
print("Training model...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.savefig(os.path.join(figures_dir, 'lorenz_loss.png'))
plt.close()

# %%
# 4. Evaluate the model
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f'Test MSE: {test_loss.item():.4f}')

# Plot results
plt.figure(figsize=(12, 6))
# Convert tensors to numpy for plotting. Flatten y_test_tensor to 1D array.
plt.plot(y_test_tensor.numpy().flatten(), label='True Z')
plt.plot(test_predictions.numpy().flatten(), label='Predicted Z', linestyle='--')
plt.title('Lorenz System Z Prediction (Test Set)')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'lorenz_prediction.png'))
plt.close()
print(f"Plots saved to {figures_dir}")
