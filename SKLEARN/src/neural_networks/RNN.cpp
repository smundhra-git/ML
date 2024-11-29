#include "SKLEARN/neural_networks/RNN.h"

void RNN::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd RNN::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd RNN::get_coefficients() const {
    return coefficients;
}

