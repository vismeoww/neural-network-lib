import numpy as np
import matplotlib.pyplot as plt
import pywrap.pynn as pynn

# 1. Generate Synthetic Polynomial Data
np.random.seed(42)
N = 150
# Create 1D input x values ranging from -2 to 2
x_np = np.linspace(-2.0, 2.0, N).reshape(N, 1).astype(np.float32)

# Define a polynomial target function: y = 0.5x^3 - 1.5x + 0.5
y_np = (0.5 * (x_np ** 3) - 1.5 * x_np + 0.5).astype(np.float32)

# 2. Initialize Graph and Architecture
g = pynn.Graph(seed=42)

# Define inputs with shape [N, 1]
x_id = g.input(pynn.Shape(N, 1), label="x")
target_id = g.input(pynn.Shape(N, 1), label="target")

# Build a small Multi-Layer Perceptron (MLP) for non-linear curve fitting
h1 = pynn.linear(g, x_id, out=32, init=pynn.Init.He)
h1_relu = pynn.relu(g, h1)

h2 = pynn.linear(g, h1_relu, out=32, init=pynn.Init.He)
h2_relu = pynn.relu(g, h2)

pred_id = pynn.linear(g, h2_relu, out=1, init=pynn.Init.He)

# Mean Squared Error Loss
loss_id = pynn.mse_loss(g, pred_id, target_id)

# 3. Setup Optimizer and Parameters
optimizer = pynn.SGD(lr=0.02, momentum=0.9)
params = g.param_pairs()

# Feed initial dataset into C++ tensors
g.value(x_id).from_numpy(x_np)
g.value(target_id).from_numpy(y_np)

# 4. Training Loop
epochs = 2500
print("Starting training...")
for epoch in range(epochs):
    g.forward()
    
    if epoch % 250 == 0:
        loss_val = g.value(loss_id).to_numpy().item()
        print(f"Epoch {epoch:4d} | Loss: {loss_val:.6f}")
        
    g.backward(loss_id)
    optimizer.step(params)

# 5. Extract Final Predictions
preds_np = g.value(pred_id).to_numpy()

# 6. Plotting Results with Matplotlib
plt.figure(figsize=(9, 5))
plt.plot(x_np, y_np, 'k--', label="Target Polynomial: $y = 0.5x^3 - 1.5x + 0.5$", linewidth=2.5)
plt.plot(x_np, preds_np, 'r-', label="C++ Engine Prediction", linewidth=2)
plt.scatter(x_np, y_np, color='blue', s=15, alpha=0.4, label="Training Samples")

plt.title("Polynomial Curve Fitting via C++ Neural Network Engine", fontsize=12)
plt.xlabel("x", fontsize=11)
plt.ylabel("y", fontsize=11)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()