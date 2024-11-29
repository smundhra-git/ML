#ifndef SKLEARN_GANS_H
#define SKLEARN_GANS_H

#include <Eigen/Dense>

class GANs {
public:
    GANs() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_GANS_H

