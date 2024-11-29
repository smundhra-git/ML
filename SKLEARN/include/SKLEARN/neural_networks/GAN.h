#ifndef SKLEARN_GAN_H
#define SKLEARN_GAN_H

#include <Eigen/Dense>

class GAN {
public:
    GAN() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_GAN_H

