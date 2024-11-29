#ifndef SKLEARN_ISOLATIONFOREST_H
#define SKLEARN_ISOLATIONFOREST_H

#include <Eigen/Dense>

class IsolationForest {
public:
    IsolationForest() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_ISOLATIONFOREST_H

