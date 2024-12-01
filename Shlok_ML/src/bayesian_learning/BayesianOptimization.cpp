#include "Shlok_ML/bayesian_learning/BayesianOptimization.h"

void BayesianOptimization::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd BayesianOptimization::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd BayesianOptimization::get_coefficients() const {
    return coefficients;
}

