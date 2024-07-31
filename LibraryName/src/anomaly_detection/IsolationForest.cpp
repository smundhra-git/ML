#include "LibraryName/anomaly_detection/IsolationForest.h"

void IsolationForest::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd IsolationForest::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd IsolationForest::get_coefficients() const {
    return coefficients;
}

