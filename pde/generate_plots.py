import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

ub = 200
rb = 20

def func(t, r):
    x, y = r
    dx_t = 1 / ub * rb * (2.0 * ub * x - 0.04 * ub * x * ub * y)
    dy_t = 1 / ub * rb * (0.02 * ub * x * ub * y - 1.06 * ub * y)
    return dx_t, dy_t

def get_true_solution(t_eval):
    # t_eval must be sorted for solve_ivp to work efficiently or at all if we want specific points
    # But solve_ivp can handle t_eval.
    # Note: initial.py used (0, 10) as span but t in [0, 1].
    sol = integrate.solve_ivp(func, (0, 1.0), (100 / ub, 15 / ub), t_eval=t_eval)
    return sol.y[0], sol.y[1]

# 1. Plot Loss History
try:
    loss_data = np.loadtxt("loss.dat")
    # Check if file is empty or has comments
    if loss_data.size > 0:
        steps = loss_data[:, 0]
        # loss.dat columns: step, train_loss_1, train_loss_2, test_loss_1, test_loss_2
        # Total train loss is sum of train_loss_1 and train_loss_2
        train_loss = np.sum(loss_data[:, 1:3], axis=1)
        test_loss = np.sum(loss_data[:, 3:5], axis=1)

        plt.figure(figsize=(10, 6))
        plt.semilogy(steps, train_loss, label="Train Loss", alpha=0.8)
        plt.semilogy(steps, test_loss, label="Test Loss", linestyle="--", alpha=0.8)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training and Testing Loss History")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig("loss_history.png")
        plt.close()
        print("Generated loss_history.png")
    else:
        print("loss.dat is empty")
except Exception as e:
    print(f"Error generating loss plot: {e}")

# 2. Plot Error Analysis & Phase Plane
try:
    test_data = np.loadtxt("test.dat")
    if test_data.size > 0:
        t_pred = test_data[:, 0]
        x_pred = test_data[:, 1]
        y_pred = test_data[:, 2]

        # Sort by t just in case
        sort_idx = np.argsort(t_pred)
        t_pred = t_pred[sort_idx]
        x_pred = x_pred[sort_idx]
        y_pred = y_pred[sort_idx]

        # Get true solution at these points
        x_true, y_true = get_true_solution(t_pred)

        # Calculate errors
        error_x = np.abs(x_true - x_pred)
        error_y = np.abs(y_true - y_pred)

        plt.figure(figsize=(10, 6))
        plt.plot(t_pred, error_x, label="Error in x(t)", color="red")
        plt.plot(t_pred, error_y, label="Error in y(t)", color="blue")
        plt.xlabel("t")
        plt.ylabel("Absolute Error")
        plt.title("Absolute Error of Predictions")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("error_analysis.png")
        plt.close()
        print("Generated error_analysis.png")

        # 3. Phase Plane Plot (x vs y)
        plt.figure(figsize=(8, 8))
        plt.plot(x_true, y_true, 'k-', label='True Orbit')
        plt.plot(x_pred, y_pred, 'r--', label='Predicted Orbit')
        plt.xlabel("Prey Population (scaled)")
        plt.ylabel("Predator Population (scaled)")
        plt.title("Phase Plane Trajectory")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("phase_plane.png")
        plt.close()
        print("Generated phase_plane.png")
    else:
        print("test.dat is empty")
except Exception as e:
    print(f"Error generating error/phase plots: {e}")
