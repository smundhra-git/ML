#ifndef SKLEARN_ORDINALREGRESSION_H
#define SKLEARN_ORDINALREGRESSION_H

#include <Eigen/Dense>

class OrdinalRegression {
public:
    OrdinalRegression() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_ORDINALREGRESSION_H

