#include "Shlok_ML/instance_based/KNN.h"

void KNN::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd KNN::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd KNN::get_coefficients() const {
    return coefficients;
}

