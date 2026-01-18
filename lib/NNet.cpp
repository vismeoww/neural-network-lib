#include "NNet.h"
#include <iostream>

/**
 * ========================================================================================================================
 * =======================================      NODE FUNCTIONS
 * ==========================================================
 * ========================================================================================================================
 */

Node::Node(Operation op, Node *lhs, Node *rhs) {
  this->isbase = false;
  this->op = op;
  this->lhs = lhs;
  this->rhs = rhs;
  switch (op) {
  case ADD:
  case SUB: {
    if (lhs->data->getRows() != rhs->data->getRows() ||
        lhs->data->getCols() != rhs->data->getCols()) {
      throw("Invalid dimensions");
    }
    this->data = new Matrix(lhs->data->getRows(), lhs->data->getCols());
    break;
  }
  case MUL: {
    if (lhs->data->getCols() != rhs->data->getRows()) {
      throw("Invalid dimensions");
    }
    this->data = new Matrix(lhs->data->getRows(), rhs->data->getCols());
    break;
  }
  default:
    throw("Invalid operation");
    break;
  }
}

Node::Node(Operation op, Node *lhs) {
  this->isbase = false;
  this->op = op;
  this->lhs = lhs;
  this->rhs = nullptr;
  switch (op) {
  case RELU:
  case SOFTMAX:
  case SIGMOID: {
    this->data = new Matrix(lhs->data->getRows(), lhs->data->getCols());
    break;
  }
  default:
    throw("Invalid operation");
    break;
  }
}

Node::Node(size_t rows, size_t cols) {
  this->isbase = true;
  this->data = new Matrix(rows, cols);
  this->lhs = nullptr;
  this->rhs = nullptr;
}

Node::~Node() {
  if (!this->isbase) {
    delete this->data;
    delete this->lhs;
    delete this->rhs;
  } else {
    delete this->data;
  }
}

void Node::print() {
  if (this->isbase) {
    this->data->print();
  } else {
    std::cout << "Operation: " << this->op << std::endl;
  }
}

size_t Node::operandCount() {
  if (this->isbase) {
    return 0;
  }
  auto op = this->op;
  if (op >= __BINARY_OPERATIONS__ && op < __UNARY_OPERATIONS__) {
    return 2;
  } else {
    return 1;
  }
}

size_t Node::recNodeCount() {
  if (this->isbase) {
    return 1;
  } else {
    size_t lhs_count = this->lhs == nullptr ? 0 : this->lhs->recNodeCount();
    size_t rhs_count = this->rhs == nullptr ? 0 : this->rhs->recNodeCount();
    return 1 + lhs_count + rhs_count;
  }
}

void Node::performCompute() {
  Node *node = this;
  switch (node->op) {
  case ADD:
    node->data->add(*node->lhs->data, *node->rhs->data);
    break;
  case MUL:
    node->data->multiply(*node->lhs->data, *node->rhs->data);
    break;
  case RELU:
    node->data->relu(*node->lhs->data);
    break;
  case SIGMOID:
    node->data->sigmoid(*node->lhs->data);
    break;
  case SOFTMAX:
    node->data->softmax(*node->lhs->data);
    break;
  default:
    throw("Invalid operation");
    break;
  }
}

/**
 * ========================================================================================================================
 * ===================================        COMPUTE GRAPH FUNCTIONS
 * ===================================================
 * ========================================================================================================================
 */

ComputeGraph::ComputeGraph() {
  result = nullptr;
  input = nullptr;
}

ComputeGraph::~ComputeGraph() { delete result; }

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