#include "Mat.h"

#include <cstddef>
#include <iostream>
#include <random>

Matrix::Matrix(size_t rows, size_t cols) {
  Rows = rows;
  Cols = cols;
  data = new double[rows * cols];
}

Matrix::~Matrix() { delete[] data; }

Matrix::Matrix(const Matrix &other) {
  Rows = other.Rows;
  Cols = other.Cols;
  data = new double[Rows * Cols];
  for (size_t i = 0; i < Rows * Cols; i++) {
    data[i] = other.data[i];
  }
}

double &Matrix::operator()(std::size_t row, std::size_t col) {
  return data[row * Cols + col];
}

const double &Matrix::operator()(std::size_t row, std::size_t col) const {
  return data[row * Cols + col];
}

void Matrix::setRandom() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  for (size_t i = 0; i < Rows * Cols; i++) {
    data[i] = dis(gen);
  }
}

void Matrix::setZero() {
  for (size_t i = 0; i < Rows * Cols; i++) {
    data[i] = 0.0;
  }
}

bool Matrix::add(const Matrix &lhs, const Matrix &rhs) {
  if (lhs.Rows != rhs.Rows || lhs.Cols != rhs.Cols) {
    return false;
  }

  for (int i = 0; i < lhs.Rows; i++) {
    for (int j = 0; j < lhs.Cols; j++) {
      (*this)(i, j) = lhs(i, j) + rhs(i, j);
    }
  }
  return true;
}

bool Matrix::multiply(const Matrix &lhs, const Matrix &rhs) {
  if (lhs.Cols != rhs.Rows) {
    return false;
  }
  if (this->Rows != lhs.Rows || this->Cols != rhs.Cols) {
    return false;
  }
  for (int i = 0; i < this->Rows; i++) {
    for (int j = 0; j < this->Cols; j++) {
      (*this)(i, j) = 0;
      for (int k = 0; k < lhs.Cols; k++) {
        (*this)(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
  return true;
}

bool Matrix::softmax(const Matrix &input) {
  if (this->Rows != input.Rows || this->Cols != input.Cols) {
    return false;
  }
  double sum = 0;
  for (int i = 0; i < input.Rows; i++) {
    for (int j = 0; j < input.Cols; j++) {
      (*this)(i, j) = std::exp(input(i, j));
      sum += (*this)(i, j);
    }
  }
  for (int i = 0; i < input.Rows; i++) {
    for (int j = 0; j < input.Cols; j++) {
      (*this)(i, j) /= sum;
    }
  }
  return true;
}

bool Matrix::sigmoid(const Matrix &input) {
  if (this->Rows != input.Rows || this->Cols != input.Cols) {
    return false;
  }
  for (int i = 0; i < input.Rows; i++) {
    for (int j = 0; j < input.Cols; j++) {
      (*this)(i, j) = 1 / (1 + std::exp(-input(i, j)));
    }
  }
  return true;
}

bool Matrix::relu(const Matrix &input) {
  if (this->Rows != input.Rows || this->Cols != input.Cols) {
    return false;
  }
  for (int i = 0; i < input.Rows; i++) {
    for (int j = 0; j < input.Cols; j++) {
      (*this)(i, j) = std::max(0.0, input(i, j));
    }
  }
  return true;
}

bool Matrix::loss(double &result, const Matrix &expected) {
  if (this->Rows != expected.Rows || this->Cols != expected.Cols) {
    return false;
  }
  result = 0;
  size_t n = 0;
  for (int i = 0; i < this->Rows; i++) {
    for (int j = 0; j < this->Cols; j++) {
      n++;
      result +=
          ((*this)(i, j) - expected(i, j)) * ((*this)(i, j) - expected(i, j));
    }
  }
  result /= n;
  return true;
}

void Matrix::print() {
  for (int i = 0; i < Rows; i++) {
    for (int j = 0; j < Cols; j++) {
      std::cout << (*this)(i, j) << " ";
    }
    std::cout << std::endl;
  }
}