#include "NNet.h"
#include "Mat.h"


int main() { 
    Node* input = new Node(1, 3);
    input->data->setRandom();
    
    Node* first_later = new Node(3,4);
    first_later->data->setRandom();

    Node* first_layer_res = new Node(Operation::MUL, input, first_later);

    Node* res = new Node(Operation::SOFTMAX, first_layer_res);

    ComputeGraph graph = ComputeGraph();
    graph.setInput(input);
    graph.setResult(res);

    graph.compute();
    graph.printRes();
    
    return 0; 
}