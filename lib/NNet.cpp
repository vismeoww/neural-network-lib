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
  if (!sorted) {
    sort();
  }
  // result->performCompute();
  for (auto node : sorted_nodes) {
    if (!node->isbase) {
      node->performCompute();
    }
  }
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
  Node *node = new Node(1, n, true);
  nodes.push_back(node);
  node->data->setZero();
  return node;
}

Node *ComputeGraph::addLayer(Node *prev, size_t layer_size, Operation op) {
  sorted = false;
  if (numOperandsFromOp(op) > 1) {
    throw("expected layer to have unary operation at the end");
  }
  Node *layer = new Node(prev->data->getCols(), layer_size, false);
  layer->data->setRandom();
  nodes.push_back(layer);
  Node *mul = new Node(Operation::MUL, prev, layer);
  Node *out = new Node(op, mul);
  nodes.push_back(out);
  return out;
}

void sortHelper(std::vector<Node *> &sorted_nodes, Node *curr) {
  // if unary operation, then we do it on the lhs
  if (curr->operandCount() == 1) {
    sortHelper(sorted_nodes, curr->lhs);
  }
  // if binary operation, then we do it on the lhs and rhs
  else if (curr->operandCount() == 2) {
    sortHelper(sorted_nodes, curr->lhs);
    sortHelper(sorted_nodes, curr->rhs);
  }
  sorted_nodes.push_back(curr);
  return;
}

void ComputeGraph::sort() {
  sorted_nodes = std::vector<Node *>();
  sortHelper(sorted_nodes, result);
  sorted = true;
}

void ComputeGraph::printSortedList() {
  if (!sorted) {
    sort();
  }
  for (auto node : sorted_nodes) {
    node->print();
  }
}
