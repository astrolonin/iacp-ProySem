#ifndef MATRIX
#define MATRIX

#include <iostream>
#include <array>
#include <iomanip>
#include <vector>
#include <cassert>

struct MatrixHCI {
    int rows; // T rows (frames)
    int cols; // D columns (pixels)
    std::vector<double> data;

    // Constructor allocating the flat vector
    MatrixHCI(int r, int c) : rows(r), cols(c), data(r*c, 0.0) {}

    // Column-major indexing
    double& operator()(int r, int c) {return data[c * rows + r];}
};

#endif
