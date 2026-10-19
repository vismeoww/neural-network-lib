#include "NNet.h"
#include "Mat.h"
#include "Node.h"


int main() { 
    ComputeGraph graph = ComputeGraph();
    Node* input = graph.createInputNode(2);
    Node* l1 = graph.addLayer(input, 4, SIGMOID);
    Node* l2 = graph.addLayer(l1, 2, SOFTMAX);
    graph.setResult(l2);
    graph.setInput(input);
    input->data->setRandom();


    graph.compute();
    graph.printRes();
    graph.printAst();
    // sort
    graph.sort();
    graph.printSortedList();
    
    return 0; 
}
