#include "matrix_hci.hpp"
#include "svd_seq.hpp"

int main(){
    int T = 6; int D = 5;

    MatrixHCI X = randMatrix(T, D);
    MatrixHCI W = X;
    printMatrix(X, "Experimento");

    MatrixHCI V(D,D);
    for (int i = 0; i < D; ++i){V(i,i) = 1.0;}

    SVDSequential(W, V);

    
    printMatrix(W, "X Ortogonal (W)");
    printMatrix(V, "Matrix V final");

    printMatrix(W.transpose()*W, "Deberia ser diagonal");
    printMatrix(W*V.transpose(), "Deberia ser X");
    
    return 0;
}
