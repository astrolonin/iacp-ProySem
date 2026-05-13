#include "matrix_hci.hpp"
#include "svd_seq.hpp"

int main(){
    int T = 4; int D = 3;

    MatrixHCI X = randMatrix(T, D);
    printMatrix(X, "Experimento");

    MatrixHCI V(D,D);
    for (int i = 0; i < D; ++i){V(i,i) = 1.0;}

    SVDSequential(X, V);

    
    printMatrix(X, "X Ortogonal (W)");
    printMatrix(V, "Matrix V final");

    printMatrix(X.transpose()*X, "Deberia ser diagonal");
    return 0;
}
