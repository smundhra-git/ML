#ifndef SKLEARN_MLP_H
#define SKLEARN_MLP_H

#include <Eigen/Dense>

class MLP {
public:
    MLP() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_MLP_H

