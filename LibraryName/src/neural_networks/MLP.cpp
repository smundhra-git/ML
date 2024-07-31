#include "LibraryName/neural_networks/MLP.h"

void MLP::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd MLP::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd MLP::get_coefficients() const {
    return coefficients;
}

