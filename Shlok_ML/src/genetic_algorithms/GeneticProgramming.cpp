#include "Shlok_ML/genetic_algorithms/GeneticProgramming.h"

void GeneticProgramming::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd GeneticProgramming::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd GeneticProgramming::get_coefficients() const {
    return coefficients;
}

