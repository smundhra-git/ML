#include "LibraryName/other/OrdinalRegression.h"

void OrdinalRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd OrdinalRegression::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd OrdinalRegression::get_coefficients() const {
    return coefficients;
}

