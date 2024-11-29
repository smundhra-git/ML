#include "SKLEARN/bayesian_learning/BayesianNetworks.h"

void BayesianNetworks::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd BayesianNetworks::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd BayesianNetworks::get_coefficients() const {
    return coefficients;
}

