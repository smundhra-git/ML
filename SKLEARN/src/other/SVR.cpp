#include "SKLEARN/other/SVR.h"

void SVR::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd SVR::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd SVR::get_coefficients() const {
    return coefficients;
}

