#include "Shlok_ML/dimensionality_reduction/SVD.h"

void SVD::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd SVD::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd SVD::get_coefficients() const {
    return coefficients;
}

