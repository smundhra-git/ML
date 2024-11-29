#include "SKLEARN/regression/RidgeRegression.h"

void RidgeRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd RidgeRegression::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd RidgeRegression::get_coefficients() const {
    return coefficients;
}

