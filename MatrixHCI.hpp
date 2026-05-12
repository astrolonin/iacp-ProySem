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

#endif
