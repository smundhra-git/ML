#include "LibraryName/reinforcement_learning/DQN.h"

void DQN::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd DQN::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd DQN::get_coefficients() const {
    return coefficients;
}

