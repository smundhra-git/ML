#ifndef LIBRARYNAME_RNN_H
#define LIBRARYNAME_RNN_H

#include <Eigen/Dense>

class RNN {
public:
    RNN() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_RNN_H

