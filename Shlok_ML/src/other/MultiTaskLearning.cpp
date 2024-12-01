#include "Shlok_ML/other/MultiTaShlok_MLing.h"

void MultiTaShlok_MLing::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd MultiTaShlok_MLing::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd MultiTaShlok_MLing::get_coefficients() const {
    return coefficients;
}

