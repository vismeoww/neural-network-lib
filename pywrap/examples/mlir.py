import pywrap.pynn.mlir as pynn
import pywrap.pynn as nn

g = pynn.MLIRGraph(seed=42)
N = 150

x_id = g.input(nn.Shape(N, 1), label="x")
target_id = g.input(nn.Shape(N, 1), label="target")

# Build a small Multi-Layer Perceptron (MLP) for non-linear curve fitting
h1 = pynn.linear(g, x_id, out=32, init=pynn.MLIRInit.He)
h1_relu = pynn.relu(g, h1)

pred_id = pynn.linear(g, h1_relu, out=1, init=pynn.MLIRInit.He)

loss_id = pynn.mse_loss(g,pred_id, target_id)

g.finalizeForward(loss_id)
g.print()