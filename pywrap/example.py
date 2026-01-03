import pynn
import numpy as np

if __name__ == "__main__":
    nn = pynn.NeuralNetwork([2, 3, 1], 0.005)
    # create some data for f(x,y) = 2x+10y
    x_range = np.linspace(0, 5, 100)
    y_range = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 2 * X + 10 * Y
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    input_data = np.column_stack((X, Y))
    output_data = Z.reshape((-1, 1))
    MAX_ITERATIONS = 1000
    for i in range(MAX_ITERATIONS):
        mse = nn.train(input_data, output_data)
        print("Iteration: ", i, "Total MSE: ", mse)