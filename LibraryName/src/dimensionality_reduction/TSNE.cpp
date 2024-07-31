#include "LibraryName/dimensionality_reduction/TSNE.h"

void TSNE::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd TSNE::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd TSNE::get_coefficients() const {
    return coefficients;
}

