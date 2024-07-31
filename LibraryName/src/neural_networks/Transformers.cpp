#include "LibraryName/neural_networks/Transformers.h"

void Transformers::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd Transformers::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd Transformers::get_coefficients() const {
    return coefficients;
}

