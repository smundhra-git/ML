#include "SKLEARN/dimensionality_reduction/PCA.h"

void PCA::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd PCA::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd PCA::get_coefficients() const {
    return coefficients;
}

