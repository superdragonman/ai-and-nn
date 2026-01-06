"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
d = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
X_train = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_train = d["y"].astype(np.float32)
d = np.load("antiderivative_aligned_test.npz", allow_pickle=True)
X_test = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_test = d["y"].astype(np.float32)

data = dde.data.TripleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# Choose a network
m = 100
dim_x = 1
net = dde.nn.DeepONetCartesianProd(
    [m, 40, 40],
    [dim_x, 40, 40],
    "relu",
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

# Plot the loss trajectory
dde.utils.plot_loss_history(losshistory)
plt.savefig("loss_history.png")
# plt.show()

# Predict and Plot additional results
y_pred = model.predict(data.test_x)

# Create a figure with 3 subplots for the first 3 test samples
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    # X_test[1] contains the coordinate points (trunk input)
    # y_test[i] is the true value for the i-th function
    # y_pred[i] is the predicted value for the i-th function
    
    # Sort X_test[1] for plotting if it's not sorted, though usually it is for plotting lines
    # Assuming 1D domain for trunk
    x_coords = X_test[1].flatten()
    sorted_indices = np.argsort(x_coords)
    x_sorted = x_coords[sorted_indices]
    y_true_sorted = y_test[i].flatten()[sorted_indices]
    y_pred_sorted = y_pred[i].flatten()[sorted_indices]
    
    plt.plot(x_sorted, y_true_sorted, 'b-', label='True')
    plt.plot(x_sorted, y_pred_sorted, 'r--', label='Predicted')
    plt.title(f'Test Sample {i+1}')
    plt.xlabel('x')
    plt.ylabel('Antiderivative')
    plt.legend()

plt.tight_layout()
plt.savefig("test_results.png")
print("Plots saved to loss_history.png and test_results.png")
