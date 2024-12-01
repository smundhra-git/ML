#include "Shlok_ML/genetic_algorithms/EvolutionaryStrategies.h"

void EvolutionaryStrategies::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd EvolutionaryStrategies::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd EvolutionaryStrategies::get_coefficients() const {
    return coefficients;
}

