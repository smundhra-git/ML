#ifndef LIBRARYNAME_BAYESIANOPTIMIZATION_H
#define LIBRARYNAME_BAYESIANOPTIMIZATION_H

#include <Eigen/Dense>

class BayesianOptimization {
public:
    BayesianOptimization() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_BAYESIANOPTIMIZATION_H

