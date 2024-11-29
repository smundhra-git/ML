#include "SKLEARN/semi_supervised/GANs.h"

void GANs::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd GANs::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd GANs::get_coefficients() const {
    return coefficients;
}

