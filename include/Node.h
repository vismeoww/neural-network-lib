#pragma once

#include "Mat.h"
#include <cstddef>
#include <string>
enum Operation {
  __BINARY_OPERATIONS__ = 0,
  ADD,
  SUB,
  MUL,
  __UNARY_OPERATIONS__,
  RELU,
  SIGMOID,
  SOFTMAX
};

size_t numOperandsFromOp(Operation op);

typedef struct Node {
  bool isbase;
  Operation op;
  std::string name;
  Matrix *data;

  // if it's a unary operation then we do it on the lhs
  // and rhs remains as nullptr
  Node *lhs;
  Node *rhs;

  // functions
  Node(Operation op, Node *lhs, Node *rhs);
  Node(Operation op, Node *lhs);
  Node(size_t rows, size_t cols);
  // named
  Node(std::string name, Operation op, Node *lhs, Node *rhs)
      : Node(op, lhs, rhs) {
    this->name = name;
  };
  Node(std::string name, Operation op, Node *lhs) : Node(op, lhs) {
    this->name = name;
  };
  Node(std::string name, size_t rows, size_t cols) : Node(rows, cols) {
    this->name = name;
  };
  ~Node();
  void print();
  void performCompute();
  size_t operandCount();
  size_t recNodeCount();
  void astPrint();
} Node;