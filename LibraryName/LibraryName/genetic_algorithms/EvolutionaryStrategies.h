#ifndef LIBRARYNAME_EVOLUTIONARYSTRATEGIES_H
#define LIBRARYNAME_EVOLUTIONARYSTRATEGIES_H

#include <Eigen/Dense>

class EvolutionaryStrategies {
public:
    EvolutionaryStrategies() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_EVOLUTIONARYSTRATEGIES_H

