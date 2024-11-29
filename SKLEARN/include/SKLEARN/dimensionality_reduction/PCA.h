#ifndef SKLEARN_PCA_H
#define SKLEARN_PCA_H

#include <Eigen/Dense>

class PCA {
public:
    PCA() = default;

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const;
    Eigen::VectorXd get_coefficients() const;

private:
    Eigen::VectorXd coefficients;
};

#endif // SKLEARN_PCA_H

