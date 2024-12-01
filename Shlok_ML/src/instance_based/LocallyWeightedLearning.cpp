#include "Shlok_ML/instance_based/LocallyWeightedLearning.h"

void LocallyWeightedLearning::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd LocallyWeightedLearning::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd LocallyWeightedLearning::get_coefficients() const {
    return coefficients;
}

