#ifndef SKLEARN_SVR_H
#define SKLEARN_SVR_H

#include <Eigen/Dense>

class SVR {
public:
    SVR() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_SVR_H

