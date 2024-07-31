#include "LibraryName/classification/RandomForest.h"

void RandomForest::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd RandomForest::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd RandomForest::get_coefficients() const {
    return coefficients;
}

