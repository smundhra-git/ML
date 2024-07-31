#include "LibraryName/regression/LinearRegression.h"

void LinearRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd LinearRegression::get_coefficients() const {
    return coefficients;
}

