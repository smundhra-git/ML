#include "Shlok_ML/reinforcement_learning/TemporalDifference.h"

void TemporalDifference::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd TemporalDifference::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd TemporalDifference::get_coefficients() const {
    return coefficients;
}

