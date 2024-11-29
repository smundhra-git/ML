#ifndef SKLEARN_STACKING_H
#define SKLEARN_STACKING_H

#include <Eigen/Dense>

class Stacking {
public:
    Stacking() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_STACKING_H

