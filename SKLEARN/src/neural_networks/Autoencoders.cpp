#include "SKLEARN/neural_networks/Autoencoders.h"

void Autoencoders::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd Autoencoders::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd Autoencoders::get_coefficients() const {
    return coefficients;
}

