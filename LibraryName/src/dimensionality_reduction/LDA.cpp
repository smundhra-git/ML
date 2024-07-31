#include "LibraryName/dimensionality_reduction/LDA.h"

void LDA::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd LDA::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd LDA::get_coefficients() const {
    return coefficients;
}

