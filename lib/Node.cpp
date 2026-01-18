#include "Node.h"
#include "Mat.h"
#include <iostream>
#include <string>

size_t numOperandsFromOp(Operation op) {
  if (op >= __BINARY_OPERATIONS__ && op < __UNARY_OPERATIONS__) {
    return 2;
  } else {
    return 1;
  }
}

std::string toString(Operation op) {
  switch (op) {
  case ADD:
    return "ADD";
  case SUB:
    return "SUB";
  case MUL:
    return "MUL";
  case RELU:
    return "RELU";
  case SIGMOID:
    return "SIGMOID";
  case SOFTMAX:
    return "SOFTMAX";
  default:
    return "UNKNOWN";
  }
}

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
    delete this->data;
    if (this->lhs != nullptr){
      delete this->lhs;
      this->lhs = nullptr;
    }

    if (this->rhs != nullptr)
    {
      delete this->rhs;
      this->rhs = nullptr;
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

void Node::astPrint() {
  if (this->isbase) {
    if (this->name == "") {

      printf("{%u x %u}", this->data->getRows(), this->data->getCols());
    } else {
      printf("{%s:(%u x %u)}", this->name.c_str(), this->data->getRows(),
             this->data->getCols());
    }
  } else {
    if (this->name == "") {
      printf("%s", toString(this->op).c_str());
    } else {
      printf("(%s, %s)", this->name.c_str(), toString(this->op).c_str());
    }
    printf("{");
    if (this->lhs != nullptr) {
      this->lhs->astPrint();
    } else {
      printf("NULL");
    }
    if (this->operandCount() > 1) {
      printf(",");
      if (this->rhs != nullptr) {
        this->rhs->astPrint();
      } else {
        printf("NULL");
      }
    }
    printf("}");
  }
}