#include "LibraryName/bayesian_learning/NaiveBayes.h"

void NaiveBayes::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd NaiveBayes::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd NaiveBayes::get_coefficients() const {
    return coefficients;
}

