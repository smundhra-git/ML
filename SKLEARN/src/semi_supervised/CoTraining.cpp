#include "SKLEARN/semi_supervised/CoTraining.h"

void CoTraining::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd CoTraining::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd CoTraining::get_coefficients() const {
    return coefficients;
}

