#pragma once

#include <cstddef>
class Matrix {
    private:
        size_t Rows;
        size_t Cols;
        double* data;
    public:
        Matrix(size_t rows, size_t cols);
        ~Matrix();
        Matrix(const Matrix& other);
        double &operator()(std::size_t row, std::size_t col);
        const double &operator()(std::size_t row, std::size_t col) const;
        size_t getRows() const { return Rows; }
        size_t getCols() const { return Cols; }
        // matrix operations
        void setRandom();
        void setZero();
        bool add(const Matrix& lhs, const Matrix& rhs);
        bool multiply(const Matrix& lhs, const Matrix& rhs);
        bool softmax(const Matrix& input);
        bool sigmoid(const Matrix& input);
        bool relu(const Matrix& input);
        bool loss(double& res, const Matrix& expected);
        // debug
        void print();

};