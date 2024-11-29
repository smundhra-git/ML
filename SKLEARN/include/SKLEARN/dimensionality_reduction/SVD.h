#ifndef SKLEARN_SVD_H
#define SKLEARN_SVD_H

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

#endif // SKLEARN_SVD_H

