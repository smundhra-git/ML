#ifndef SKLEARN_TRANSFORMERS_H
#define SKLEARN_TRANSFORMERS_H

#include <Eigen/Dense>

class Transformers {
public:
    Transformers() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_TRANSFORMERS_H

