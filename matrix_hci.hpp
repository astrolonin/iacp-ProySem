#ifndef MATRIX
#define MATRIX

#include <iostream>
#include <array>
#include <iomanip>
#include <random>
#include <vector>
#include <string>

struct MatrixHCI {
    int rows; // T rows (frames)
    int cols; // D columns (pixels)
    std::vector<double> data;

    // Constructor allocating the flat vector
    MatrixHCI(int r, int c) : rows(r), cols(c), data(r*c, 0.0) {}

    // Column-major indexing
    double& operator()(int r, int c) {return data[c * rows + r];}

    // Column-major indexing
    const double& operator()(int r, int c) const {return data[c * rows + r];}

    MatrixHCI operator*(const MatrixHCI& other) const{
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions mismatched.");
        }
        MatrixHCI result(rows, other.cols);

        for (int r = 0; r < rows; ++r){
            for (int c = 0; c < other.cols; ++c){
                double sum = 0.0;
                for (int k = 0; k < cols; ++k){
                    sum += (*this)(r,k) * other(k,c);
                }
                result(r,c) = sum;
            }
        }
        return result;
    }

    MatrixHCI transpose() const {
        MatrixHCI result(cols, rows);
        for (int r = 0; r < rows; ++r){
            for (int c = 0; c < cols; ++c){
                result(c,r) = (*this)(r,c);
            }
        }
        return result;
    }
};

void printMatrix(const MatrixHCI& m, const std::string& name){
    std::cout << "--- " << name << " (" << m.rows << "x" << m.cols << ") ---" << std::endl;
    for (int r = 0; r < m.rows; ++r){
        for (int c = 0; c < m.cols; ++c){
            std::cout << std::setw(8) << std::setprecision(4) << std::fixed << m(r,c) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;   
}

MatrixHCI randMatrix(int rows, int cols){
    MatrixHCI m(rows, cols);

    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis (-10.0,10);

    for (int r = 0; r < rows; ++r){
        for (int c = 0; c < cols; ++c){
            m(r, c) = dis(gen);
        }
    }
    return m;
}

#endif
