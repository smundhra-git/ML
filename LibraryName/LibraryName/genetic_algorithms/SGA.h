#ifndef LIBRARYNAME_SGA_H
#define LIBRARYNAME_SGA_H

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

#endif // LIBRARYNAME_SGA_H
