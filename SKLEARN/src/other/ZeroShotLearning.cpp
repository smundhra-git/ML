#include "SKLEARN/other/ZeroShotLearning.h"

void ZeroShotLearning::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd ZeroShotLearning::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd ZeroShotLearning::get_coefficients() const {
    return coefficients;
}

