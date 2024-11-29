#ifndef SKLEARN_DIFFERENTIALEVOLUTION_H
#define SKLEARN_DIFFERENTIALEVOLUTION_H

#include <Eigen/Dense>

class DifferentialEvolution {
public:
    DifferentialEvolution() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_DIFFERENTIALEVOLUTION_H

