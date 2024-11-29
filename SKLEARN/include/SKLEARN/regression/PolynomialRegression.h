#ifndef SKLEARN_POLYNOMIALREGRESSION_H
#define SKLEARN_POLYNOMIALREGRESSION_H

#include <Eigen/Dense>

class PolynomialRegression {
public:
    PolynomialRegression() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_POLYNOMIALREGRESSION_H

