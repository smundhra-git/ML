#include "SKLEARN/anomaly_detection/OneClassSVM.h"

void OneClassSVM::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd OneClassSVM::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd OneClassSVM::get_coefficients() const {
    return coefficients;
}

