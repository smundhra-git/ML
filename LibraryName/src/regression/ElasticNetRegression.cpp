#include "LibraryName/regression/ElasticNetRegression.h"

void ElasticNetRegression::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd ElasticNetRegression::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd ElasticNetRegression::get_coefficients() const {
    return coefficients;
}

