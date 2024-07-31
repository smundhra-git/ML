#include "LibraryName/reinforcement_learning/MonteCarlo.h"

void MonteCarlo::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd MonteCarlo::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd MonteCarlo::get_coefficients() const {
    return coefficients;
}

