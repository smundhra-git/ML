#include "Shlok_ML/ensemble_learning/Boosting.h"

void Boosting::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Eigen::VectorXd Boosting::predict(const Eigen::MatrixXd &X) const {
    return X * coefficients;
}

Eigen::VectorXd Boosting::get_coefficients() const {
    return coefficients;
}

