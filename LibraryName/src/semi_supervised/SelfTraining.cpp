#include "LibraryName/semi_supervised/SelfTraining.h"

void SelfTraining::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd SelfTraining::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd SelfTraining::get_coefficients() const {
    return coefficients;
}

