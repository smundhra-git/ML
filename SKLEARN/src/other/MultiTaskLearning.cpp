#include "SKLEARN/other/MultiTaskLearning.h"

void MultiTaskLearning::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd MultiTaskLearning::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd MultiTaskLearning::get_coefficients() const {
    return coefficients;
}

