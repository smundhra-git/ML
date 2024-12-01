#include "Shlok_ML/neural_networks/CNN.h"

void CNN::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd CNN::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd CNN::get_coefficients() const {
    return coefficients;
}

