#ifndef SKLEARN_BAYESIANNETWORKS_H
#define SKLEARN_BAYESIANNETWORKS_H

#include <Eigen/Dense>

class BayesianNetworks {
public:
    BayesianNetworks() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_BAYESIANNETWORKS_H

