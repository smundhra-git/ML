#ifndef LIBRARYNAME_GRADIENTBOOSTING_H
#define LIBRARYNAME_GRADIENTBOOSTING_H

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

#endif // LIBRARYNAME_GRADIENTBOOSTING_H

