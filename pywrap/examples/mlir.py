import pywrap.pynn as pynn

g = pynn.MLIRGraph(seed=42)
N = 150

x_id = g.input(pynn.Shape(N, 1), label="x")
target_id = g.input(pynn.Shape(N, 1), label="target")

# Build a small Multi-Layer Perceptron (MLP) for non-linear curve fitting
h1 = pynn.linear(g, x_id, out=32, init=pynn.Init.He)
h1_relu = pynn.relu(g, h1)

pred_id = pynn.linear(g, h1_relu, out=1, init=pynn.Init.He)

g.finalizeForward(pred_id)
g.print()