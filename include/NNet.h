#pragma once

#include "Node.h"
#include <vector>

class ComputeGraph {
private:
  Node *result;
  Node *input;
  std::vector<Node*> nodes;
  // topological sort 
  bool sorted = false;
  std::vector<Node*> sorted_nodes;

public:
  ComputeGraph();
  ~ComputeGraph();
  bool setResult(Node *node);
  bool setInput(Node *node);
  const Node *getResult() const;
  bool compute();
  // create nodes 
  Node* createInputNode(size_t n);
  Node* addLayer(Node* prev, size_t layer_size, Operation op);
  // topological sort 
  void sort();
  // debugging
  void printAst();
  void printRes();
  void printSortedList();
};
