#ifndef LIBRARYNAME_ZEROSHOTLEARNING_H
#define LIBRARYNAME_ZEROSHOTLEARNING_H

#include <Eigen/Dense>

class ZeroShotLearning {
public:
    ZeroShotLearning() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_ZEROSHOTLEARNING_H
