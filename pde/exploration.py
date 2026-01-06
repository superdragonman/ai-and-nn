import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import torch
import os

ub = 200
rb = 20

def func(t, r):
    x, y = r
    dx_t = 1 / ub * rb * (2.0 * ub * x - 0.04 * ub * x * ub * y)
    dy_t = 1 / ub * rb * (0.02 * ub * x * ub * y - 1.06 * ub * y)
    return dx_t, dy_t

def gen_truedata():
    t = np.linspace(0, 1, 100)
    sol = integrate.solve_ivp(func, (0, 10), (100 / ub, 15 / ub), t_eval=t)
    x_true, y_true = sol.y
    x_true = x_true.reshape(100, 1)
    y_true = y_true.reshape(100, 1)
    return t, x_true, y_true

def ode_system(x, y):
    r = y[:, 0:1]
    p = y[:, 1:2]
    dr_t = dde.grad.jacobian(y, x, i=0)
    dp_t = dde.grad.jacobian(y, x, i=1)
    return [
        dr_t - 1 / ub * rb * (2.0 * ub * r - 0.04 * ub * r * ub * p),
        dp_t - 1 / ub * rb * (0.02 * r * ub * p * ub - 1.06 * p * ub),
    ]

def input_transform(t):
    return torch.cat([torch.sin(t)], dim=1)

def output_transform(t, y):
    y1 = y[:, 0:1]
    y2 = y[:, 1:2]
    return torch.cat([y1 * torch.tanh(t) + 100 / ub, y2 * torch.tanh(t) + 15 / ub], dim=1)

def run_experiment(exp_name, layer_size, num_domain):
    print(f"Running experiment: {exp_name}")
    print(f"Layer size: {layer_size}, Num domain: {num_domain}")
    
    geom = dde.geometry.TimeDomain(0, 1.0)
    data = dde.data.PDE(geom, ode_system, [], num_domain, 2, num_test=1000)
    
    net = dde.nn.FNN(layer_size, "tanh", "Glorot normal")
    net.apply_feature_transform(input_transform)
    net.apply_output_transform(output_transform)
    
    model = dde.Model(data, net)
    
    # Train with Adam
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(iterations=20000) # Reduced iterations for speed in exploration
    
    # Train with L-BFGS
    model.compile("L-BFGS")
    losshistory, train_state = model.train()
    
    # Predict and calculate error
    t_true, x_true, y_true = gen_truedata()
    t_pred = t_true.reshape(100, 1)
    sol_pred = model.predict(t_pred)
    x_pred = sol_pred[:, 0:1]
    y_pred = sol_pred[:, 1:2]
    
    l2_error_x = np.linalg.norm(x_true - x_pred) / np.linalg.norm(x_true)
    l2_error_y = np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
    
    print(f"Experiment {exp_name} finished. L2 Error X: {l2_error_x:.4e}, L2 Error Y: {l2_error_y:.4e}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t_true, x_true, 'k-', label='True x')
    plt.plot(t_true, y_true, 'b-', label='True y')
    plt.plot(t_true, x_pred, 'r--', label='Pred x')
    plt.plot(t_true, y_pred, 'g--', label='Pred y')
    plt.title(f"Results for {exp_name}\nL2 Error X: {l2_error_x:.2e}, Y: {l2_error_y:.2e}")
    plt.legend()
    plt.savefig(f"exp_{exp_name}.png")
    plt.close()
    
    return l2_error_x, l2_error_y

if __name__ == "__main__":
    experiments = [
        ("Baseline", [1] + [64] * 6 + [2], 3000),
        ("Shallow", [1] + [64] * 3 + [2], 3000),
        ("Narrow", [1] + [32] * 6 + [2], 3000),
        ("Sparse", [1] + [64] * 6 + [2], 500),
        ("Dense", [1] + [64] * 6 + [2], 6000),
    ]
    
    results = []
    for name, layers, points in experiments:
        err_x, err_y = run_experiment(name, layers, points)
        results.append((name, err_x, err_y))
        
    # Save results to text file
    with open("exploration_results.txt", "w") as f:
        f.write("Experiment, L2_Error_X, L2_Error_Y\n")
        for name, err_x, err_y in results:
            f.write(f"{name}, {err_x:.4e}, {err_y:.4e}\n")
            
    # Plot comparison bar chart
    names = [r[0] for r in results]
    errs_x = [r[1] for r in results]
    errs_y = [r[2] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, errs_x, width, label='L2 Error X')
    plt.bar(x + width/2, errs_y, width, label='L2 Error Y')
    plt.ylabel('Relative L2 Error')
    plt.title('Comparison of Different Configurations')
    plt.xticks(x, names)
    plt.legend()
    plt.yscale('log')
    plt.savefig("exp_comparison.png")
    plt.close()
