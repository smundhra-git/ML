#include "SKLEARN/dimensionality_reduction/ICA.h"

void ICA::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd ICA::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd ICA::get_coefficients() const {
    return coefficients;
}

