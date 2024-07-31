#include "LibraryName/neural_networks/LSTM.h"

void LSTM::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd LSTM::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd LSTM::get_coefficients() const {
    return coefficients;
}

