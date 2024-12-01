#include "Shlok_ML/dimensionality_reduction/UMAP.h"

void UMAP::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd UMAP::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd UMAP::get_coefficients() const {
    return coefficients;
}

