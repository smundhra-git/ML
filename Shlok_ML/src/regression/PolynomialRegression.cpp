#include "Shlok_ML/regression/PolynomialRegression.h"

void PolynomialRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd PolynomialRegression::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd PolynomialRegression::get_coefficients() const {
    return coefficients;
}

