#include <iostream>

#include "MLIR/MLIRGraph.hpp"
#include "MLIR/Node.hpp"

using namespace std;
using namespace mlg;
int main(){
    cout << "Starting" <<endl;
    unsigned int B = 32;
    Graph g(42);
    NodeId x = g.input({B, 1}, "x");
    NodeId t = g.input({B, 1}, "target");
    NodeId h1 = relu(g, linear(g, x, 32, MLIRInit::Xavier));
    NodeId y = linear(g, h1, 1, MLIRInit::Xavier);
    NodeId loss = mseLoss(g, y, t);
    g.finalizeForward(loss);
    g.print();

    return 0;
}