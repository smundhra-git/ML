#ifndef LIBRARYNAME_MONTECARLO_H
#define LIBRARYNAME_MONTECARLO_H

#include <Eigen/Dense>

class MonteCarlo {
public:
    MonteCarlo() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_MONTECARLO_H

