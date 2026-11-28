#include <iostream>

#include "MLIR/MLIRGraph.hpp"
#include "MLIR/Node.hpp"
#include "MLIR/Optimizer.hpp"

using namespace std;
using namespace mlg;

static double truePoly(double x)
{
    return 0.5 * x * x * x - 0.8 * x * x + 0.3 * x + 0.1;
}

int main()
{

    // ---- data ----------------------------------------------------------------
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> ux(-1.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.02);

    const size_t N = 512, B = 64;
    std::vector<double> X(N), Y(N);
    for (size_t i = 0; i < N; i++)
    {
        X[i] = ux(rng);
        Y[i] = truePoly(X[i]) + noise(rng);
    }

    cout << "Starting" << endl;
    // unsigned int B = 32;
    Graph g(42);
    NodeId x = g.input({B, 1}, "x");
    NodeId t = g.input({B, 1}, "target");
    NodeId h1 = relu(g, linear(g, x, 32, MLIRInit::Xavier));
    NodeId y = linear(g, h1, 1, MLIRInit::Xavier);
    NodeId loss = mseLoss(g, y, t);
    g.finalizeForward({y,loss});
    g.compileBackwards(loss);
    g.print();

    auto params = g.paramPairs();
    Adam opt(0.01);

    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    const int epochs = 800;

    for (int e = 1; e <= epochs; e++)
    {
        std::shuffle(idx.begin(), idx.end(), rng);
        double epochLoss = 0;
        size_t batches = 0;
        for (size_t s = 0; s + B <= N; s += B)
        {
            for (size_t i = 0; i < B; i++)
            {
                g.value(x)(i, 0) = X[idx[s + i]];
                g.value(t)(i, 0) = Y[idx[s + i]];
            }
            g.forward();
            g.backward(loss);
            opt.step(params);
            epochLoss += g.value(loss).v[0];
            batches++;
        }
        if (e == 1 || e % 50 == 0)
            std::printf("  epoch %4d  loss %.6f\n", e, epochLoss / batches);
    }

    return 0;
}