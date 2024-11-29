#ifndef SKLEARN_LASSOREGRESSION_H
#define SKLEARN_LASSOREGRESSION_H

#include <Eigen/Dense>

class LassoRegression {
public:
    LassoRegression() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_LASSOREGRESSION_H

