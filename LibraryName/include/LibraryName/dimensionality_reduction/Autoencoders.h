#ifndef LIBRARYNAME_AUTOENCODERS_H
#define LIBRARYNAME_AUTOENCODERS_H

#include <Eigen/Dense>

class Autoencoders {
public:
    Autoencoders() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_AUTOENCODERS_H

