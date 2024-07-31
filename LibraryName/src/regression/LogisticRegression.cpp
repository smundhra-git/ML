#include "LibraryName/regression/LogisticRegression.h"

void LogisticRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd LogisticRegression::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd LogisticRegression::get_coefficients() const {
    return coefficients;
}

