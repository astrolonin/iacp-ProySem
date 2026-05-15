#include "matrix_hci.hpp"
#include "svd_seq.hpp"

int main(){
    int T = 6; int D = 5;

    MatrixHCI X = randMatrix(T, D);
    MatrixHCI W = X;
    printMatrix(X, "X");

    MatrixHCI V(D,D);
    for (int i = 0; i < D; ++i){V(i,i) = 1.0;}

    SVDSequential(W, V);

    printMatrix(W, "W");
    printMatrix(V, "V");

    MatrixHCI U(T,D); std::vector<double> Sigma;

    extractUSigma(W,U,Sigma);

    std::cout << "Singular Values (Sigma):\n";
    for(int i = 0; i < D; ++i) {
        std::cout << "Sigma[" << i << "] = " << std::fixed << std::setprecision(4) << Sigma[i] << "\n";
    }
    std::cout << "\n";

    printMatrix(U.transpose()*U, "Deberia ser la matriz identidad");
    
    return 0;
}
