#ifndef SKLEARN_GRADIENTBOOSTING_H
#define SKLEARN_GRADIENTBOOSTING_H

#include <Eigen/Dense>

class GradientBoosting {
public:
    GradientBoosting() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_GRADIENTBOOSTING_H

