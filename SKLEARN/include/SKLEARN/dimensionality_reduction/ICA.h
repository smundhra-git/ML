#ifndef SKLEARN_ICA_H
#define SKLEARN_ICA_H

#include <Eigen/Dense>

class ICA {
public:
    ICA() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_ICA_H
