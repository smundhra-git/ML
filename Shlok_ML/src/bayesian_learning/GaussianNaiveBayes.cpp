#include "Shlok_ML/bayesian_learning/GaussianNaiveBayes.h"

void GaussianNaiveBayes::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd GaussianNaiveBayes::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd GaussianNaiveBayes::get_coefficients() const {
    return coefficients;
}

