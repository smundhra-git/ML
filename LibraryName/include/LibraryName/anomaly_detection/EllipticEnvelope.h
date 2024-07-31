#ifndef LIBRARYNAME_ELLIPTICENVELOPE_H
#define LIBRARYNAME_ELLIPTICENVELOPE_H

#include <Eigen/Dense>

class EllipticEnvelope {
public:
    EllipticEnvelope() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_ELLIPTICENVELOPE_H

