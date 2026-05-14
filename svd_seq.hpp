#include "matrix_hci.hpp"

template <typename T>
int sign(T val){
    return (T(0) < val) - (val < T(0));
    //return (val >= T(0) ? 1 : -1);
}

void SVDSequential(MatrixHCI& X, MatrixHCI& V, double epsilon = 1e-7){
    int T = X.rows, D = X.cols;
    bool any_rots = true;
    int sweeps = 0, max_sweeps = 100000;

    while (any_rots && sweeps < max_sweeps){
        any_rots = false;
        for (int i = 0; i < D-1; ++i){
            for (int j = 0; j < D; ++j){
                 
                double a = 0.0, b = 0.0, c = 0.0;
                for (int r = 0; r < T; ++r){
                    a += X(r, i) * X(r, i);
                    b += X(r, j) * X(r, j);
                    c += X(r, i) * X(r, j); 
                }
                
                if (a < 1e-12|| b < 1e-12) continue;
                double corr = std::abs(c) / std::sqrt(a*b);

                if (corr > epsilon){
                    any_rots = true;

                    double tau = (b - a) / (2.0 * c);
                    double t = sign(tau) / (std::abs(tau) + std::sqrt(1.0 + tau*tau));
                    double c_rot = 1.0 / std::sqrt(1.0 + t * t);
                    double s_rot = c_rot * t;

                    for (int r = 0; r < T; ++r){
                        double x_i = X(r, i);
                        double x_j = X(r, j);
                        X(r, i) = c_rot * x_i - s_rot * x_j;
                        X(r, j) = s_rot * x_i + c_rot * x_j; 
                    }

                    for (int r = 0; r < D; ++r){
                        double v_i = V(r, i);
                        double v_j = V(r, j);
                        V(r, i) =  c_rot * v_i - s_rot * v_j;
                        V(r, j) =  s_rot * v_i + c_rot * v_j;
                    }
                }
            }
        }
        sweeps++;
    }
    std::cout << "Converged in " << sweeps << " sweeps.\n\n";
}