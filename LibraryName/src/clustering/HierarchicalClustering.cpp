#include "LibraryName/clustering/HierarchicalClustering.h"

void HierarchicalClustering::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd HierarchicalClustering::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd HierarchicalClustering::get_coefficients() const {
    return coefficients;
}

