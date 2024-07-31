#include "LibraryName/classification/GradientBoosting.h"

void GradientBoosting::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd GradientBoosting::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd GradientBoosting::get_coefficients() const {
    return coefficients;
}

