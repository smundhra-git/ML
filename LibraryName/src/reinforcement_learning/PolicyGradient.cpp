#include "LibraryName/reinforcement_learning/PolicyGradient.h"

void PolicyGradient::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd PolicyGradient::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd PolicyGradient::get_coefficients() const {
    return coefficients;
}

