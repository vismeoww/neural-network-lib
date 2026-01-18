#include "NNet.h"
#include "Node.h"
#include <cstdio>
#include <iostream>

ComputeGraph::ComputeGraph() {
  nodes = std::vector<Node *>();
  result = nullptr;
  input = nullptr;
}

ComputeGraph::~ComputeGraph() { 
  delete result; 
  result = nullptr;
}

bool ComputeGraph::compute() {
  result->performCompute();
  return true;
}

bool ComputeGraph::setResult(Node *node) {
  if (result != nullptr) {
    return false;
  }
  result = node;
  return true;
}

bool ComputeGraph::setInput(Node *node) {
  if (input != nullptr) {
    return false;
  }
  input = node;
  return true;
}

const Node *ComputeGraph::getResult() const { return result; }

void ComputeGraph::printRes() { result->data->print(); }

void ComputeGraph::printAst() {
  result->astPrint();
  printf("\n");
}

Node *ComputeGraph::createInputNode(size_t n) {
  Node* node = new Node(1, n);
  nodes.push_back(node);
  node->data->setZero();
  return node;
}

Node* ComputeGraph::addLayer(Node* prev, size_t layer_size, Operation op) {
  if (numOperandsFromOp(op)>1){
    throw("expected layer to have unary operation at the end");
  }
  Node* layer = new Node(prev->data->getCols(), layer_size);
  layer->data->setRandom();
  nodes.push_back(layer);
  Node* mul = new Node(Operation::MUL, prev, layer);
  Node* out = new Node(op, mul);
  nodes.push_back(out);
  return out;
}