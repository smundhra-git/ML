#include "LibraryName/ensemble_learning/Stacking.h"

void Stacking::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd Stacking::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd Stacking::get_coefficients() const {
    return coefficients;
}

