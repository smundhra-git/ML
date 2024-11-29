#include "SKLEARN/clustering/DBSCAN.h"

void DBSCAN::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd DBSCAN::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd DBSCAN::get_coefficients() const {
    return coefficients;
}

