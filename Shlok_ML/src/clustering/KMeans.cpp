#include "Shlok_ML/clustering/KMeans.h"

void KMeans::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd KMeans::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd KMeans::get_coefficients() const {
    return coefficients;
}

