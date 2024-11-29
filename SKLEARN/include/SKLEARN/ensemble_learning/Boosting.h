#ifndef SKLEARN_BOOSTING_H
#define SKLEARN_BOOSTING_H

#include <Eigen/Dense>

class Boosting {
public:
    Boosting() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_BOOSTING_H

