#ifndef LIBRARYNAME_SVD_H
#define LIBRARYNAME_SVD_H

#include <Eigen/Dense>

class SVD {
public:
    SVD() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // LIBRARYNAME_SVD_H

