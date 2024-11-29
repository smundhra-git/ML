#ifndef SKLEARN_SGA_H
#define SKLEARN_SGA_H

#include <Eigen/Dense>

class SGA {
public:
    SGA() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_SGA_H

