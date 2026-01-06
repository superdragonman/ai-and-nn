"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure figures directory exists
if not os.path.exists("figures"):
    os.makedirs("figures")

def load_data():
    d = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
    X_train = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
    y_train = d["y"].astype(np.float32)
    d = np.load("antiderivative_aligned_test.npz", allow_pickle=True)
    X_test = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
    y_test = d["y"].astype(np.float32)
    return X_train, y_train, X_test, y_test

def run_experiment(name, X_train, y_train, X_test, y_test, net_arch):
    print(f"\nRunning Experiment: {name}")
    
    data = dde.data.TripleCartesianProd(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    net = dde.nn.DeepONetCartesianProd(
        net_arch["branch"],
        net_arch["trunk"],
        "relu",
        "Glorot normal",
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
    
    losshistory, train_state = model.train(iterations=10000)
    
    # Save loss plot
    dde.utils.plot_loss_history(losshistory)
    plt.title(f"Loss History - {name}")
    plt.savefig(f"figures/loss_{name}.png")
    plt.close()

    # Evaluate on test set
    # We want to evaluate on the FULL test set always
    # The model.train() already evaluates on the provided X_test/y_test in 'data'
    # So we can just take the final metric
    
    final_train_loss = train_state.loss_train[-1]
    final_test_loss = train_state.loss_test[-1]
    final_metric = train_state.metrics_test[-1]
    
    print(f"Results for {name}:")
    print(f"  Train Loss: {final_train_loss}")
    print(f"  Test Loss: {final_test_loss}")
    print(f"  Test Metric: {final_metric}")
    
    return {
        "name": name,
        "train_loss": final_train_loss,
        "test_loss": final_test_loss,
        "metric": final_metric,
        "model": model,
        "losshistory": losshistory
    }

def main():
    X_train_orig, y_train_orig, X_test, y_test = load_data()
    
    results = []
    
    # Experiment 1: Baseline
    # Branch: [100, 40, 40], Trunk: [1, 40, 40]
    # Full sampling (100 points)
    exp1_arch = {"branch": [100, 40, 40], "trunk": [1, 40, 40]}
    res1 = run_experiment("Baseline", X_train_orig, y_train_orig, X_test, y_test, exp1_arch)
    results.append(res1)
    
    # Experiment 2: Deeper Architecture
    # Branch: [100, 40, 40, 40], Trunk: [1, 40, 40, 40]
    exp2_arch = {"branch": [100, 40, 40, 40], "trunk": [1, 40, 40, 40]}
    res2 = run_experiment("Deeper_Net", X_train_orig, y_train_orig, X_test, y_test, exp2_arch)
    results.append(res2)
    
    # Experiment 3: Sparse Sampling (20 points)
    # Subsample X_train[1] and y_train
    # X_train[1] is (100, 1), y_train is (150, 100)
    indices = np.linspace(0, 99, 20, dtype=int)
    X_train_sparse = (X_train_orig[0], X_train_orig[1][indices])
    y_train_sparse = y_train_orig[:, indices]
    
    # Use Baseline architecture
    res3 = run_experiment("Sparse_Sampling", X_train_sparse, y_train_sparse, X_test, y_test, exp1_arch)
    results.append(res3)
    
    # Save results to a text file for easy reading
    with open("exploration_results.txt", "w") as f:
        f.write("Experiment Results\n")
        f.write("==================\n")
        for res in results:
            f.write(f"Experiment: {res['name']}\n")
            f.write(f"  Train Loss: {res['train_loss']}\n")
            f.write(f"  Test Loss: {res['test_loss']}\n")
            f.write(f"  Test Metric (L2 Rel Error): {res['metric']}\n")
            f.write("-" * 30 + "\n")

    # --- New Plots ---

    # 1. Combined Loss History Plot
    plt.figure(figsize=(10, 6))
    for res in results:
        lh = res['losshistory']
        steps = lh.steps
        loss_train = np.array(lh.loss_train).sum(axis=1) # Sum losses if multiple, though here likely 1
        plt.semilogy(steps, loss_train, label=f"{res['name']} Train Loss")
    plt.title("Training Loss Comparison")
    plt.xlabel("Iterations")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig("figures/loss_comparison.png")
    plt.close()

    # 2. Multi-sample Prediction Comparison
    # Plot 3 samples
    plt.figure(figsize=(15, 5))
    sample_indices = [0, 10, 20] # Pick 3 distinct samples
    
    x_coords = X_test[1].flatten()
    sorted_indices = np.argsort(x_coords)
    x_sorted = x_coords[sorted_indices]

    for i, idx in enumerate(sample_indices):
        plt.subplot(1, 3, i + 1)
        y_true = y_test[idx].flatten()[sorted_indices]
        plt.plot(x_sorted, y_true, 'k-', linewidth=2, label='True')
        
        colors = ['r--', 'g--', 'b--']
        for j, res in enumerate(results):
            model = res['model']
            # We need to predict again or store predictions. Predicting is fast enough.
            y_pred_all = model.predict(X_test)
            y_pred_sample = y_pred_all[idx].flatten()[sorted_indices]
            plt.plot(x_sorted, y_pred_sample, colors[j], label=res['name'], alpha=0.8)
            
        plt.title(f"Test Sample {idx}")
        plt.xlabel("x")
        if i == 0:
            plt.ylabel("Antiderivative")
        plt.legend(fontsize='small')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/prediction_comparison_multi.png")
    plt.close()

    # 3. Error Distribution Plot
    plt.figure(figsize=(10, 6))
    for res in results:
        model = res['model']
        y_pred = model.predict(X_test)
        # Calculate relative L2 error for each sample
        # y_test shape: (N_samples, N_points)
        # y_pred shape: (N_samples, N_points)
        
        # L2 norm per sample
        diff_norm = np.linalg.norm(y_test - y_pred, axis=1)
        true_norm = np.linalg.norm(y_test, axis=1)
        rel_error = diff_norm / (true_norm + 1e-8) # Avoid division by zero
        
        plt.hist(rel_error, bins=30, alpha=0.5, label=res['name'], density=True)
        
    plt.title("Error Distribution (Relative L2 Error)")
    plt.xlabel("Relative L2 Error")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("figures/error_distribution.png")
    plt.close()

    # 4. Absolute Error Comparison for Sample 0
    # Plot the absolute error |y_true - y_pred| for the first sample across the domain
    plt.figure(figsize=(10, 6))
    
    sample_idx = 0
    x_coords = X_test[1].flatten()
    sorted_indices = np.argsort(x_coords)
    x_sorted = x_coords[sorted_indices]
    y_true = y_test[sample_idx].flatten()[sorted_indices]
    
    for res in results:
        model = res['model']
        y_pred = model.predict(X_test)
        y_pred_sample = y_pred[sample_idx].flatten()[sorted_indices]
        abs_error = np.abs(y_true - y_pred_sample)
        plt.plot(x_sorted, abs_error, label=f"{res['name']} Error")
        
    plt.title(f"Absolute Error along Domain (Sample {sample_idx})")
    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig("figures/absolute_error_sample.png")
    plt.close()

    print("All plots saved to figures/")


if __name__ == "__main__":
    main()
