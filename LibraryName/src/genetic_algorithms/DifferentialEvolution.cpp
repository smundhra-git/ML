#include "LibraryName/genetic_algorithms/DifferentialEvolution.h"

void DifferentialEvolution::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd DifferentialEvolution::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd DifferentialEvolution::get_coefficients() const {
    return coefficients;
}

