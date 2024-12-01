#include "Shlok_ML/genetic_algorithms/SGA.h"

void SGA::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd SGA::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd SGA::get_coefficients() const {
    return coefficients;
}

