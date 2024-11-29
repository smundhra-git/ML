#include "SKLEARN/clustering/MeanShift.h"

void MeanShift::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd MeanShift::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd MeanShift::get_coefficients() const {
    return coefficients;
}

