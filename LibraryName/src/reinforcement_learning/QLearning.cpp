#include "LibraryName/reinforcement_learning/QLearning.h"

void QLearning::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd QLearning::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd QLearning::get_coefficients() const {
    return coefficients;
}

