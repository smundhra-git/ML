#include "SKLEARN/anomaly_detection/LOF.h"

void LOF::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd LOF::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd LOF::get_coefficients() const {
    return coefficients;
}

