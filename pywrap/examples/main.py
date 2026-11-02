import pywrap.pynn as pynn
import numpy as np

# 1. Initialize Graph and define architecture
g = pynn.Graph(seed=42)

# Input placeholder (e.g., batch_size=32, input_dim=10)
x_id = g.input(pynn.Shape(32, 10), label="features")
target_id = g.input(pynn.Shape(32, 1), label="targets")

# Hidden layers using your C++ helpers
h1 = pynn.linear(g, x_id, out=16, init=pynn.Init.He)
h1_relu = pynn.relu(g, h1)
pred_id = pynn.linear(g, h1_relu, out=1, init=pynn.Init.He)

# Loss
loss_id = pynn.mse_loss(g, pred_id, target_id)

# 2. Setup Optimizer
optimizer = pynn.SGD(lr=0.01, momentum=0.9)
params = g.param_pairs()

# 3. Dummy Training Loop with NumPy Data
for epoch in range(100):
    # Generate random batch data via NumPy
    x_data = np.random.randn(32, 10).astype(np.float32)
    y_data = np.random.randn(32, 1).astype(np.float32)

    # Feed NumPy data into C++ input tensors
    g.value(x_id).from_numpy(x_data)
    g.value(target_id).from_numpy(y_data)

    # Forward pass
    g.forward()

    # Compute loss value in Python if desired
    loss_val = g.value(loss_id).to_numpy()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss_val.item():.4f}")

    # Backward pass starting from loss node
    g.backward(loss_id)

    # Optimizer step
    optimizer.step(params)