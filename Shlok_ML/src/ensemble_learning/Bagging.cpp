#include "Shlok_ML/ensemble_learning/Bagging.h"

void Bagging::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd Bagging::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd Bagging::get_coefficients() const {
    return coefficients;
}

