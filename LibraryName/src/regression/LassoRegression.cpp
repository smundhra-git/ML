#include "LibraryName/regression/LassoRegression.h"

void LassoRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd LassoRegression::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd LassoRegression::get_coefficients() const {
    return coefficients;
}

